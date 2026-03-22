/**
 * LiteRT-LM Node.js Bindings
 *
 * TypeScript wrapper providing the same API as mcp-llm Rust crate
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

// @ts-ignore - Native addon
const native = require(join(__dirname, 'litert_lm_node.node'));

/**
 * Backend type for model execution
 */
export enum LiteRTBackend {
  /** CPU backend */
  CPU = 'cpu',
  /** GPU backend (if available) */
  GPU = 'gpu',
}

/**
 * Benchmark data for a single turn (prefill or decode)
 */
export interface TurnBenchmark {
  /** Number of tokens processed */
  numTokens: number;
  /** Time taken in seconds */
  durationSeconds: number;
  /** Throughput (tokens/second) */
  tokensPerSec: number;
}

/**
 * Complete benchmark information for a conversation
 */
export interface BenchmarkInfo {
  /** All prefill turns */
  prefillTurns: TurnBenchmark[];
  /** All decode turns */
  decodeTurns: TurnBenchmark[];
  /** Time to first token in milliseconds */
  timeToFirstTokenMs: number;
  /** Convenience: last prefill token count */
  lastPrefillTokenCount: number;
  /** Convenience: last decode token count */
  lastDecodeTokenCount: number;
}

/**
 * Message role in conversation
 */
export type MessageRole = 'user' | 'model' | 'system' | 'tool';

/**
 * Tool parameter property definition
 */
export interface ToolParameterProperty {
  type: string;
  description?: string;
  enum?: string[];
}

/**
 * Tool definition for function calling (matches LiteRT-LM tools_json format)
 */
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, ToolParameterProperty>;
    required?: string[];
  };
}

/**
 * A parsed function call extracted from model output
 */
export interface ToolCall {
  name: string;
  arguments: Record<string, string>;
}

/**
 * Result of sending a message that may contain tool calls
 */
export interface SendMessageResult {
  /** The raw text response from the model */
  text: string;
  /** Parsed tool calls, if the model requested function invocations */
  toolCalls: ToolCall[];
  /** Whether the response contains tool calls that need handling */
  hasToolCalls: boolean;
}

/**
 * Parse function calls from model output using Gemma 3n function call tokens.
 *
 * Gemma 3n / FunctionGemma format:
 *   <start_function_call>call:function_name{param1:<escape>value1<escape>,param2:<escape>value2<escape>}<end_function_call>
 *
 * Also handles the JSON-based tool_calls format from LiteRT-LM:
 *   {"tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]}
 */
export function parseToolCalls(text: string): ToolCall[] {
  const calls: ToolCall[] = [];

  // Pattern 1: Gemma-style <start_function_call>...<end_function_call>
  const gemmaPattern = /<start_function_call>\s*call:(\w+)\{([^}]*)\}\s*<end_function_call>/g;
  let match: RegExpExecArray | null;

  while ((match = gemmaPattern.exec(text)) !== null) {
    const name = match[1];
    const argsStr = match[2];
    const args: Record<string, string> = {};

    if (argsStr.trim()) {
      // Parse key:<escape>value<escape> pairs
      const paramPattern = /(\w+):<escape>(.*?)<escape>/g;
      let paramMatch: RegExpExecArray | null;
      while ((paramMatch = paramPattern.exec(argsStr)) !== null) {
        args[paramMatch[1]] = paramMatch[2];
      }

      // Fallback: parse key:value pairs without <escape> tags
      if (Object.keys(args).length === 0) {
        const simplePairs = argsStr.split(',');
        for (const pair of simplePairs) {
          const colonIdx = pair.indexOf(':');
          if (colonIdx > 0) {
            const key = pair.slice(0, colonIdx).trim();
            const value = pair.slice(colonIdx + 1).trim();
            args[key] = value;
          }
        }
      }
    }

    calls.push({ name, arguments: args });
  }

  // Pattern 2: JSON tool_calls format (LiteRT-LM native)
  if (calls.length === 0) {
    try {
      const parsed = JSON.parse(text);
      if (parsed?.tool_calls && Array.isArray(parsed.tool_calls)) {
        for (const tc of parsed.tool_calls) {
          if (tc.function?.name) {
            calls.push({
              name: tc.function.name,
              arguments: tc.function.arguments ?? {},
            });
          }
        }
      }
    } catch {
      // Not JSON — that's fine, no tool calls found
    }
  }

  return calls;
}

/**
 * Opaque conversation handle from native code
 */
type ConversationHandle = unknown;

/**
 * LiteRT-LM Conversation
 *
 * Manages a conversation session with the model
 */
export class LiteRTConversation {
  private handle: ConversationHandle;
  private destroyed: boolean = false;

  /** @internal */
  constructor(handle: ConversationHandle) {
    this.handle = handle;
  }

  /**
   * Send a message to the conversation
   *
   * @param role - Message role ('user', 'model', 'system', or 'tool')
   * @param content - Message text content
   * @returns The model's response text
   */
  sendMessage(role: MessageRole, content: string): string {
    if (this.destroyed) {
      throw new Error('Conversation has been destroyed');
    }
    return native.conversationSendMessage(this.handle, role, content);
  }

  /**
   * Send a message and automatically parse any tool calls in the response.
   *
   * Use this with conversations created via createConversationWithTools()
   * to get structured tool call data from Gemma 3n function-calling output.
   *
   * @param role - Message role
   * @param content - Message text content
   * @returns Parsed result with text, toolCalls array, and hasToolCalls flag
   */
  sendMessageForToolUse(role: MessageRole, content: string): SendMessageResult {
    const text = this.sendMessage(role, content);
    const toolCalls = parseToolCalls(text);
    return {
      text,
      toolCalls,
      hasToolCalls: toolCalls.length > 0,
    };
  }

  /**
   * Send a tool/function result back to the model.
   *
   * After receiving a tool call from sendMessageForToolUse(), execute the
   * function and send the result back using this method.
   *
   * @param toolName - Name of the tool that was called
   * @param result - The result object to send back as JSON
   * @returns The model's follow-up response (parsed for additional tool calls)
   */
  sendToolResult(toolName: string, result: Record<string, unknown>): SendMessageResult {
    const content = JSON.stringify({ tool_name: toolName, ...result });
    return this.sendMessageForToolUse('tool', content);
  }

  /**
   * Get benchmark information for this conversation
   *
   * @returns Benchmark data, or null if benchmarking is not enabled
   */
  getBenchmarkInfo(): BenchmarkInfo | null {
    if (this.destroyed) {
      throw new Error('Conversation has been destroyed');
    }
    return native.conversationGetBenchmarkInfo(this.handle);
  }

  /**
   * Destroy the conversation and free resources
   */
  destroy(): void {
    if (!this.destroyed) {
      native.conversationDestroy(this.handle);
      this.destroyed = true;
    }
  }
}

/**
 * LiteRT-LM Engine
 *
 * Main interface to the LiteRT-LM library. Create an engine to load a model,
 * then create conversations to interact with it.
 */
export class LiteRTEngine {
  private native: any;
  private destroyed: boolean = false;
  private backend!: LiteRTBackend;

  /**
   * Create a new LiteRT-LM Engine
   *
   * @param modelPath - Path to the .litertlm model file
   * @param backend - Backend to use ('cpu' or 'gpu')
   */
  constructor(modelPath: string, backend: LiteRTBackend = LiteRTBackend.GPU) {
    const backendsToTry =
      backend === LiteRTBackend.GPU
        ? [LiteRTBackend.GPU, LiteRTBackend.CPU]
        : [backend];

    let lastError: unknown = null;

    for (const candidate of backendsToTry) {
      try {
        this.native = new native.LiteRTEngine(modelPath, candidate);
        this.backend = candidate;
        return;
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError ?? new Error('Failed to initialize LiteRTEngine');
  }

  /**
   * Create a new conversation with default config
   *
   * @returns A new conversation instance
   */
  createConversation(): LiteRTConversation {
    if (this.destroyed) {
      throw new Error('Engine has been destroyed');
    }
    const handle = this.native.createConversation();
    return new LiteRTConversation(handle);
  }

  /**
   * Create a new conversation with a system instruction
   *
   * @param systemInstruction - System instruction to guide the model's behavior
   * @returns A new conversation instance
   */
  createConversationWithSystem(systemInstruction: string): LiteRTConversation {
    if (this.destroyed) {
      throw new Error('Engine has been destroyed');
    }
    const handle = this.native.createConversationWithSystem(systemInstruction);
    return new LiteRTConversation(handle);
  }

  /**
   * Create a new conversation with tool/function calling support.
   *
   * This enables Gemma 3n function-calling mode using the LiteRT-LM
   * CreateWithTools API. The model will generate structured function calls
   * using <start_function_call>/<end_function_call> tokens when it
   * determines a tool should be invoked.
   *
   * @param tools - Array of tool definitions declaring available functions
   * @param systemInstruction - Optional system instruction
   * @returns A new conversation instance configured for function calling
   */
  createConversationWithTools(
    tools: ToolDefinition[],
    systemInstruction?: string,
  ): LiteRTConversation {
    if (this.destroyed) {
      throw new Error('Engine has been destroyed');
    }
    const toolsJson = JSON.stringify(tools);
    const handle = this.native.createConversationWithTools(
      systemInstruction ?? null,
      toolsJson,
    );
    return new LiteRTConversation(handle);
  }

  /**
   * Destroy the engine and free resources
   */
  destroy(): void {
    if (!this.destroyed) {
      this.native.destroy();
      this.destroyed = true;
    }
  }

  /**
   * Returns the backend currently in use by the engine.
   */
  getBackend(): LiteRTBackend {
    return this.backend;
  }
}

/**
 * Response format options (matching mcp-llm API)
 */
export enum ResponseFormat {
  Text = 'text',
  Json = 'json',
}

// Re-export for convenience
export { LiteRTBackend as Backend };

/**
 * Create a LiteRT-LM engine
 *
 * @param modelPath - Path to the model file
 * @param backend - Backend to use
 * @returns A new LiteRTEngine instance
 */
export function createEngine(
  modelPath: string,
  backend: LiteRTBackend = LiteRTBackend.GPU
): LiteRTEngine {
  return new LiteRTEngine(modelPath, backend);
}

// Default export
export default {
  LiteRTEngine,
  LiteRTConversation,
  LiteRTBackend,
  ResponseFormat,
  createEngine,
  parseToolCalls,
};
