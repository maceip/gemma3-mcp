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
export type MessageRole = 'user' | 'model' | 'system';

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
   * @param role - Message role ('user', 'model', or 'system')
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
};
