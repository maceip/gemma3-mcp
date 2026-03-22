/**
 * Tool call parsing for Gemma 3n / FunctionGemma function-calling output.
 *
 * Standalone module — no native addon dependency, can be used and tested
 * independently of the LiteRT-LM runtime.
 */

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
 * Parse function calls from model output.
 *
 * Supports three output formats:
 *
 * 1. Gemma 3n stock format (```tool_code blocks):
 *    ```tool_code
 *    {"tool_name": "fn", "parameters": {...}}
 *    ```
 *    Also handles {"name": "fn", "parameters": {...}} variant.
 *
 * 2. FunctionGemma format (special tokens):
 *    <start_function_call>call:fn{param:<escape>val<escape>}<end_function_call>
 *
 * 3. LiteRT-LM JSON format:
 *    {"tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]}
 */
export function parseToolCalls(text: string): ToolCall[] {
  const calls: ToolCall[] = [];

  // Pattern 1: Gemma 3n stock ```tool_code blocks
  // This is what the unmodified Gemma 3n model actually outputs
  const toolCodePattern = /```tool_code\s*\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;

  while ((match = toolCodePattern.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1].trim());
      const name = parsed.tool_name ?? parsed.name;
      const args = parsed.parameters ?? parsed.arguments ?? {};
      if (name) {
        // Convert all arg values to strings for consistency
        const stringArgs: Record<string, string> = {};
        for (const [k, v] of Object.entries(args)) {
          stringArgs[k] = String(v);
        }
        calls.push({ name, arguments: stringArgs });
      }
    } catch {
      // Malformed JSON in tool_code block — skip
    }
  }

  if (calls.length > 0) return calls;

  // Pattern 2: FunctionGemma-style <start_function_call>...<end_function_call>
  const gemmaPattern = /<start_function_call>\s*call:(\w+)\{([^}]*)\}\s*<end_function_call>/g;

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

  if (calls.length > 0) return calls;

  // Pattern 3: JSON tool_calls format (LiteRT-LM native)
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

  return calls;
}
