/**
 * Gemma 3n Function Calling Example
 *
 * Demonstrates using the LiteRT-LM CreateWithTools API to enable
 * function calling with Gemma 3n models. This applies the same
 * technique Google uses for FunctionGemma (fine-tuned 270M model)
 * but targets the larger Gemma 3n E2B/E4B models instead.
 *
 * The model generates structured function calls using special tokens:
 *   <start_function_call>call:function_name{args}<end_function_call>
 *
 * Usage:
 *   bun examples/function-calling.ts /path/to/gemma-3n.litertlm [cpu|gpu]
 */

import {
  LiteRTEngine,
  LiteRTBackend,
  type ToolDefinition,
  type ToolCall,
} from '../src/index';
import * as readline from 'readline';

// Define available tools using the LiteRT-LM tool schema format
const tools: ToolDefinition[] = [
  {
    name: 'get_weather',
    description: 'Get the current weather for a given location.',
    parameters: {
      type: 'object',
      properties: {
        location: {
          type: 'string',
          description: 'City name, e.g. "San Francisco"',
        },
        unit: {
          type: 'string',
          description: 'Temperature unit',
          enum: ['celsius', 'fahrenheit'],
        },
      },
      required: ['location'],
    },
  },
  {
    name: 'search_contacts',
    description: 'Search for a contact by name in the address book.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Name or partial name to search for',
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'send_message',
    description: 'Send a text message to a contact.',
    parameters: {
      type: 'object',
      properties: {
        recipient: {
          type: 'string',
          description: 'Name of the recipient',
        },
        message: {
          type: 'string',
          description: 'The message text to send',
        },
      },
      required: ['recipient', 'message'],
    },
  },
  {
    name: 'set_alarm',
    description: 'Set an alarm for a specific time.',
    parameters: {
      type: 'object',
      properties: {
        time: {
          type: 'string',
          description: 'Time in HH:MM format (24-hour)',
        },
        label: {
          type: 'string',
          description: 'Optional label for the alarm',
        },
      },
      required: ['time'],
    },
  },
];

// Mock tool implementations
function executeTool(call: ToolCall): Record<string, unknown> {
  switch (call.name) {
    case 'get_weather':
      return {
        location: call.arguments.location,
        temperature: 72,
        unit: call.arguments.unit ?? 'fahrenheit',
        condition: 'Partly cloudy',
      };
    case 'search_contacts':
      return {
        results: [
          { name: 'John Smith', phone: '+1-555-0123' },
          { name: 'Jane Smith', phone: '+1-555-0456' },
        ],
      };
    case 'send_message':
      return {
        status: 'sent',
        recipient: call.arguments.recipient,
        timestamp: new Date().toISOString(),
      };
    case 'set_alarm':
      return {
        status: 'set',
        time: call.arguments.time,
        label: call.arguments.label ?? 'Alarm',
      };
    default:
      return { error: `Unknown tool: ${call.name}` };
  }
}

// --- Main ---

const modelPath = process.argv[2];
if (!modelPath) {
  console.error('Usage: bun examples/function-calling.ts <model.litertlm> [cpu|gpu]');
  process.exit(1);
}

const backendArg = (process.argv[3] ?? 'cpu').toLowerCase();
const backend = backendArg === 'gpu' ? LiteRTBackend.GPU : LiteRTBackend.CPU;

console.log(`Loading model: ${modelPath}`);
console.log(`Backend: ${backend}`);
console.log(`Tools: ${tools.map((t) => t.name).join(', ')}`);
console.log('---');

const engine = new LiteRTEngine(modelPath, backend);
const conversation = engine.createConversationWithTools(tools, 'You are a helpful assistant that can call functions to help the user.');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function prompt() {
  rl.question('\nYou: ', async (input) => {
    const trimmed = input.trim();
    if (!trimmed || trimmed === 'exit' || trimmed === 'quit') {
      console.log('Goodbye!');
      conversation.destroy();
      engine.destroy();
      rl.close();
      return;
    }

    try {
      let result = conversation.sendMessageForToolUse('user', trimmed);
      console.log(`\nModel: ${result.text}`);

      // Tool call loop: execute tools and feed results back until
      // the model produces a natural language response
      while (result.hasToolCalls) {
        for (const call of result.toolCalls) {
          console.log(`\n  [Tool Call] ${call.name}(${JSON.stringify(call.arguments)})`);
          const toolResult = executeTool(call);
          console.log(`  [Tool Result] ${JSON.stringify(toolResult)}`);
          result = conversation.sendToolResult(call.name, toolResult);
          console.log(`\nModel: ${result.text}`);
        }
      }

      // Show benchmark if available
      const bench = conversation.getBenchmarkInfo();
      if (bench) {
        const lastDecode = bench.decodeTurns[bench.decodeTurns.length - 1];
        if (lastDecode) {
          console.log(`  [${lastDecode.tokensPerSec.toFixed(1)} tok/s, TTFT: ${bench.timeToFirstTokenMs.toFixed(0)}ms]`);
        }
      }
    } catch (err) {
      console.error('Error:', err);
    }

    prompt();
  });
}

console.log('Type a message to test function calling. Try:');
console.log('  "What\'s the weather in Paris?"');
console.log('  "Set an alarm for 7:30"');
console.log('  "Send a message to John saying hello"');
console.log('Type "exit" to quit.\n');
prompt();
