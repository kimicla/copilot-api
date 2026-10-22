import { describe, expect, test } from "bun:test"

import type { ChatCompletionChunk, ChatCompletionsPayload } from "../src/services/copilot/create-chat-completions"

import { translateToOpenAI, translateToAnthropic } from "../src/routes/messages/non-stream-translation"
import { translateChunkToAnthropicEvents } from "../src/routes/messages/stream-translation"
import type { AnthropicStreamState } from "../src/routes/messages/anthropic-types"
import {
  chatToResponsesPayload,
  responsesEventsToChatEvents,
  responsesToChatCompletion,
} from "../src/services/copilot/create-responses-compat"

const model = "gpt-5.6-luna"

describe("Copilot Responses compatibility", () => {
  test("converts Claude tool calls and tool results into Responses input", () => {
    const anthropic = {
      model,
      max_tokens: 256,
      system: "Use tools when needed.",
      messages: [
        { role: "user" as const, content: "list files" },
        { role: "assistant" as const, content: [
          { type: "tool_use" as const, id: "call_123", name: "list_files", input: { path: "." } },
        ] },
        { role: "user" as const, content: [
          { type: "tool_result" as const, tool_use_id: "call_123", content: "README.md" },
        ] },
      ],
      tools: [{
        name: "list_files",
        description: "List directory contents",
        input_schema: { type: "object", properties: { path: { type: "string" } } },
      }],
      tool_choice: { type: "auto" as const },
    }

    const input = chatToResponsesPayload(translateToOpenAI(anthropic))
    expect(input.model).toBe(model)
    expect(input.max_output_tokens).toBe(256)
    expect(input.instructions).toBe("Use tools when needed.")
    expect(input.input).toEqual([
      { type: "message", role: "user", content: [{ type: "input_text", text: "list files" }] },
      { type: "function_call", call_id: "call_123", name: "list_files", arguments: '{"path":"."}' },
      { type: "function_call_output", call_id: "call_123", output: "README.md" },
    ])
    expect(input.tools).toEqual([{
      type: "function",
      name: "list_files",
      description: "List directory contents",
      parameters: { type: "object", properties: { path: { type: "string" } } },
    }])
  })

  test("keeps short user IDs unchanged for Responses requests", () => {
    const payload: ChatCompletionsPayload = {
      model,
      user: "claude-session-123",
      messages: [{ role: "user", content: "hi" }],
    }
    expect(chatToResponsesPayload(payload).user).toBe("claude-session-123")
  })

  test("hashes Claude Code user IDs that exceed the 64-character limit", () => {
    const longUserId = "session-".repeat(19)
    expect(longUserId.length).toBeGreaterThan(64)

    const payload: ChatCompletionsPayload = {
      model,
      user: longUserId,
      messages: [{ role: "user", content: "hi" }],
    }
    const first = chatToResponsesPayload(payload).user
    const second = chatToResponsesPayload(payload).user
    expect(first).toBe(second)
    expect(typeof first).toBe("string")
    expect(first).toMatch(/^[a-f0-9]{64}$/)
    expect(first).not.toBe(longUserId)
  })

  test("normalizes non-ASCII user IDs by UTF-8 length", () => {
    const payload: ChatCompletionsPayload = {
      model,
      user: "测试用户".repeat(20),
      messages: [{ role: "user", content: "hi" }],
    }
    expect(chatToResponsesPayload(payload).user).toMatch(/^[a-f0-9]{64}$/)
  })

  test("converts image messages without stripping media", () => {
    const payload: ChatCompletionsPayload = {
      model, messages: [{ role: "user", content: [
        { type: "text", text: "describe this" },
        { type: "image_url", image_url: { url: "data:image/png;base64,QQ==" } },
      ] }],
    }
    expect(chatToResponsesPayload(payload).input).toEqual([
      { type: "message", role: "user", content: [
        { type: "input_text", text: "describe this" },
        { type: "input_image", image_url: "data:image/png;base64,QQ==" },
      ] },
    ])
  })

  test("returns Anthropic tool_use and preserves cached usage", () => {
    const result = responsesToChatCompletion({
      id: "resp_123",
      model,
      status: "completed",
      output: [{
        type: "function_call",
        call_id: "call_123",
        name: "list_files",
        arguments: '{"path":"."}',
      }],
      usage: {
        input_tokens: 70,
        output_tokens: 10,
        input_tokens_details: { cached_tokens: 20 },
      },
    }, model)
    const anthropic = translateToAnthropic(result)
    expect(anthropic.stop_reason).toBe("tool_use")
    expect(anthropic.content).toEqual([
      { type: "tool_use", id: "call_123", name: "list_files", input: { path: "." } },
    ])
    expect(anthropic.usage.input_tokens).toBe(50)
    expect(anthropic.usage.cache_read_input_tokens).toBe(20)
  })

  test("streams text, tool JSON, and terminal stop event to Claude Code", async () => {
    const source = async function* () {
      for (const item of [
        { type: "response.created", response: { id: "resp_1", model } },
        { type: "response.output_text.delta", delta: "Looking up" },
        { type: "response.output_item.added", output_index: 1, item: {
          type: "function_call", call_id: "call_2", name: "read_file", arguments: "",
        } },
        { type: "response.function_call_arguments.delta", output_index: 1, delta: '{"name":' },
        { type: "response.function_call_arguments.delta", output_index: 1, delta: '"README.md"}' },
        { type: "response.output_item.done", output_index: 1, item: {
          type: "function_call", call_id: "call_2", name: "read_file", arguments: '{"name":"README.md"}',
        } },
        { type: "response.completed", response: {
          id: "resp_1", model, status: "completed",
          usage: { input_tokens: 100, output_tokens: 15 },
        } },
      ]) yield { data: JSON.stringify(item) }
    }

    const chatEvents: ChatCompletionChunk[] = []
    const anthropicEvents: string[] = []
    const state: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      toolCalls: {},
    }
    for await (const raw of responsesEventsToChatEvents(source(), model)) {
      if (raw.data === "[DONE]") continue
      const chunk = JSON.parse(raw.data) as ChatCompletionChunk
      chatEvents.push(chunk)
      for (const event of translateChunkToAnthropicEvents(chunk, state)) {
        anthropicEvents.push(event.type)
        if (event.type === "content_block_delta") anthropicEvents.push(event.delta.type)
      }
    }
    expect(chatEvents.at(-1)?.choices).toBeDefined()
    expect(anthropicEvents).toContain("message_start")
    expect(anthropicEvents).toContain("input_json_delta")
    expect(anthropicEvents).toContain("message_stop")
    expect(chatEvents.at(-1)?.choices?.[0]?.finish_reason).toBe("tool_calls")
  })

  test("rejects truncated SSE responses instead of reporting success", async () => {
    const source = async function* () {
      yield { data: JSON.stringify({ type: "response.output_text.delta", delta: "partial" }) }
    }
    let error = ""
    try {
      for await (const _ of responsesEventsToChatEvents(source(), model)) {
        // Drain stream
      }
    } catch (caught) {
      error = String(caught)
    }
    expect(error).toContain("before completion")
  })
})
