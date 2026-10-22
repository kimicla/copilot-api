import { randomUUID } from "node:crypto"

import { events } from "fetch-event-stream"

import { copilotBaseUrl, copilotHeaders } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"

import type {
  ChatCompletionChunk,
  ChatCompletionResponse,
  ChatCompletionsPayload,
  ContentPart,
  Message,
} from "./create-chat-completions"

type JsonObject = Record<string, unknown>

type ResponseItem = {
  type?: string
  call_id?: string
  name?: string
  arguments?: string
  content?: Array<{ type?: string; text?: string }>
}

type CopilotResponse = {
  id?: string
  model?: string
  status?: string
  output?: ResponseItem[]
  usage?: {
    input_tokens?: number
    output_tokens?: number
    total_tokens?: number
    input_tokens_details?: { cached_tokens?: number }
  }
  incomplete_details?: { reason?: string }
  error?: { message?: string }
}

type ResponseEvent = {
  type?: string
  response?: CopilotResponse
  output_index?: number
  item?: ResponseItem
  delta?: string
  arguments?: string
  error?: { message?: string }
}

// Chat Completions input is already produced by the project's Anthropic translator.
// Map it to Responses input without losing tool call IDs or image blocks.
function contentParts(
  content: Message["content"],
  role: "user" | "assistant",
): JsonObject[] {
  if (typeof content === "string") {
    return [{ type: role === "assistant" ? "output_text" : "input_text", text: content }]
  }
  if (!content) return []
  return content.flatMap<JsonObject>((part: ContentPart) => {
    if (part.type === "text") {
      return [{ type: role === "assistant" ? "output_text" : "input_text", text: part.text }]
    }
    if (part.type === "image_url") {
      return [{ type: "input_image", image_url: part.image_url.url }]
    }
    return []
  })
}

export function chatToResponsesPayload(payload: ChatCompletionsPayload): JsonObject {
  const instructions: string[] = []
  const input: JsonObject[] = []

  for (const message of payload.messages) {
    if (message.role === "system" || message.role === "developer") {
      const text = contentParts(message.content, "user")
        .map((part) => String(part.text ?? ""))
        .join("\n")
      if (text) instructions.push(text)
    } else if (message.role === "tool") {
      input.push({
        type: "function_call_output",
        call_id: message.tool_call_id,
        output: typeof message.content === "string"
          ? message.content
          : contentParts(message.content, "user")
            .map((part) => String(part.text ?? ""))
            .join("\n"),
      })
    } else {
      const blocks = contentParts(message.content, message.role)
      if (blocks.length > 0) {
        input.push({ type: "message", role: message.role, content: blocks })
      }
      if (message.role === "assistant") {
        for (const tool of message.tool_calls ?? []) {
          input.push({
            type: "function_call",
            call_id: tool.id,
            name: tool.function.name,
            arguments: tool.function.arguments,
          })
        }
      }
    }
  }

  const result: JsonObject = {
    model: payload.model,
    input,
  }
  if (instructions.length) result.instructions = instructions.join("\n\n")
  if (payload.stream) result.stream = true
  if (typeof payload.max_tokens === "number") {
    result.max_output_tokens = Math.max(16, payload.max_tokens)
  }
  if (payload.user) result.user = payload.user

  // Reasoning models commonly reject temperature, top_p, and stop.
  // These Chat Completions options are intentionally not sent to Responses.
  if (payload.tools?.length) {
    result.tools = payload.tools.map((tool) => ({
      type: "function",
      name: tool.function.name,
      description: tool.function.description ?? "",
      parameters: tool.function.parameters,
    }))
  }
  if (payload.tool_choice) {
    result.tool_choice =
      typeof payload.tool_choice === "object"
        ? { type: "function", name: payload.tool_choice.function.name }
        : payload.tool_choice
  }
  return result
}

function completionUsage(usage?: CopilotResponse["usage"]) {
  const cached = usage?.input_tokens_details?.cached_tokens
  return {
    prompt_tokens: usage?.input_tokens ?? 0,
    completion_tokens: usage?.output_tokens ?? 0,
    total_tokens:
      usage?.total_tokens ?? (usage?.input_tokens ?? 0) + (usage?.output_tokens ?? 0),
    ...(cached !== undefined ? { prompt_tokens_details: { cached_tokens: cached } } : {}),
  }
}

function finishReason(response: CopilotResponse, hasTools: boolean) {
  if (response.status === "incomplete") {
    return response.incomplete_details?.reason === "max_output_tokens"
      ? "length" as const
      : "content_filter" as const
  }
  return hasTools ? "tool_calls" as const : "stop" as const
}

export function responsesToChatCompletion(
  response: CopilotResponse,
  requestedModel: string,
): ChatCompletionResponse {
  if (response.status === "failed") {
    throw new Error(response.error?.message ?? "Copilot Responses request failed")
  }

  const text: string[] = []
  const toolCalls: Array<{
    id: string
    type: "function"
    function: { name: string; arguments: string }
  }> = []
  for (const item of response.output ?? []) {
    if (item.type === "message") {
      for (const part of item.content ?? []) {
        if (part.type === "output_text" && part.text) text.push(part.text)
      }
    } else if (item.type === "function_call") {
      toolCalls.push({
        id: item.call_id ?? randomUUID(),
        type: "function",
        function: {
          name: item.name ?? "",
          arguments: item.arguments || "{}",
        },
      })
    }
  }

  return {
    id: response.id ?? randomUUID(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: response.model ?? requestedModel,
    choices: [{
      index: 0,
      message: {
        role: "assistant",
        content: text.join("") || null,
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      },
      logprobs: null,
      finish_reason: finishReason(response, toolCalls.length > 0),
    }],
    usage: completionUsage(response.usage),
  }
}

function chatChunk(
  id: string,
  model: string,
  delta: ChatCompletionChunk["choices"][number]["delta"],
  reason: ChatCompletionChunk["choices"][number]["finish_reason"] = null,
  usage?: CopilotResponse["usage"],
): ChatCompletionChunk {
  return {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta, finish_reason: reason, logprobs: null }],
    ...(usage ? { usage: completionUsage(usage) } : {}),
  }
}

// Translate upstream Responses SSE events into the Chat Completions SSE
// events consumed by the existing Anthropic stream translator.
export async function* responsesEventsToChatEvents(
  upstream: AsyncIterable<{ data?: string; event?: string }>,
  requestedModel: string,
): AsyncGenerator<{ data: string }> {
  let id = randomUUID()
  let model = requestedModel
  let started = false
  let toolIndex = 0
  let usedTools = false
  let finished = false
  const pending = new Map<number, ResponseItem>()

  function make(delta: ChatCompletionChunk["choices"][number]["delta"],
    reason: ChatCompletionChunk["choices"][number]["finish_reason"] = null,
    usage?: CopilotResponse["usage"]) {
    return { data: JSON.stringify(chatChunk(id, model, delta, reason, usage)) }
  }

  for await (const raw of upstream) {
    if (!raw.data || raw.data === "[DONE]") continue

    let event: ResponseEvent
    try {
      event = JSON.parse(raw.data) as ResponseEvent
    } catch {
      continue
    }
    const type = event.type ?? raw.event
    if (event.response) {
      id = event.response.id ?? id
      model = event.response.model ?? model
    }

    if (!started) {
      started = true
      yield make({ role: "assistant" })
    }

    if (type === "response.output_text.delta") {
      if (event.delta) yield make({ content: event.delta })
    } else if (type === "response.output_item.added") {
      if (event.item?.type === "function_call" && event.output_index !== undefined) {
        pending.set(event.output_index, { ...event.item })
      }
    } else if (type === "response.function_call_arguments.delta") {
      if (event.output_index === undefined) continue
      const tool = pending.get(event.output_index)
      if (tool && event.delta) {
        tool.arguments = (tool.arguments ?? "") + event.delta
      }
    } else if (type === "response.function_call_arguments.done") {
      if (event.output_index === undefined) continue
      const tool = pending.get(event.output_index)
      if (tool && typeof event.arguments === "string") tool.arguments = event.arguments
    } else if (type === "response.output_item.done") {
      if (event.item?.type !== "function_call") continue
      const index = event.output_index
      const saved = index === undefined ? undefined : pending.get(index)
      const tool = { ...saved, ...event.item }
      tool.arguments = event.item.arguments || saved?.arguments || "{}"
      if (index !== undefined) pending.delete(index)
      usedTools = true
      const currentIndex = toolIndex++
      yield make({
        tool_calls: [{
          index: currentIndex,
          id: tool.call_id ?? randomUUID(),
          type: "function",
          function: { name: tool.name ?? "", arguments: "" },
        }],
      })
      yield make({
        tool_calls: [{
          index: currentIndex,
          function: { arguments: tool.arguments ?? "{}" },
        }],
      })
    } else if (type === "response.failed" || type === "response.error" || type === "error") {
      throw new Error(
        event.response?.error?.message ?? event.error?.message ?? "Copilot Responses stream failed",
      )
    } else if (type === "response.completed" || type === "response.incomplete") {
      // Some upstream implementations omit output_item.done; finalize pending calls.
      for (const tool of pending.values()) {
        usedTools = true
        const currentIndex = toolIndex++
        yield make({
          tool_calls: [{
            index: currentIndex,
            id: tool.call_id ?? randomUUID(),
            type: "function",
            function: { name: tool.name ?? "", arguments: "" },
          }],
        })
        yield make({
          tool_calls: [{
            index: currentIndex,
            function: { arguments: tool.arguments ?? "{}" },
          }],
        })
      }
      pending.clear()
      const response = event.response ?? {}
      const reason = finishReason(
        type === "response.incomplete" ? { ...response, status: "incomplete" } : response,
        usedTools,
      )
      yield make({}, reason, response.usage)
      finished = true
      break
    }
  }

  if (!finished) {
    throw new Error("Copilot Responses stream ended before completion")
  }
  yield { data: "[DONE]" }
}

export async function createResponsesAsChat(
  payload: ChatCompletionsPayload,
): Promise<ChatCompletionResponse | AsyncGenerator<{ data: string }>> {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const hasImage = payload.messages.some((message) =>
    Array.isArray(message.content) && message.content.some((part) => part.type === "image_url"),
  )
  const isAgentCall = payload.messages.some(
    (message) => message.role === "assistant" || message.role === "tool",
  )

  const response = await fetch(copilotBaseUrl(state) + "/responses", {
    method: "POST",
    headers: {
      ...copilotHeaders(state, hasImage),
      "accept-encoding": "identity",
      "X-Initiator": isAgentCall ? "agent" : "user",
    },
    body: JSON.stringify(chatToResponsesPayload(payload)),
  })

  if (!response.ok) throw new HTTPError("Failed to create responses", response)

  if (payload.stream) {
    return responsesEventsToChatEvents(events(response), payload.model)
  }
  return responsesToChatCompletion(
    (await response.json()) as CopilotResponse,
    payload.model,
  )
}
