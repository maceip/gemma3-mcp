/**
 * Basic tests for LiteRT-LM Node.js bindings
 */

import { test, expect, describe } from 'bun:test';
import {
  LiteRTEngine,
  LiteRTBackend,
  LiteRTConversation,
  parseToolCalls,
  type ToolDefinition,
} from '../src/index';

describe('LiteRT-LM Node.js Bindings', () => {
  const TEST_MODEL_PATH = process.env.LITERT_TEST_MODEL || '';

  test('should export main classes', () => {
    expect(LiteRTEngine).toBeDefined();
    expect(LiteRTConversation).toBeDefined();
    expect(LiteRTBackend).toBeDefined();
  });

  test('should have correct backend enum values', () => {
    expect(LiteRTBackend.CPU).toBe('cpu');
    expect(LiteRTBackend.GPU).toBe('gpu');
  });

  // Skip model tests if no test model is configured
  if (!TEST_MODEL_PATH) {
    test.skip('Model tests require LITERT_TEST_MODEL env var', () => {});
    return;
  }

  test('should create engine with CPU backend', () => {
    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    expect(engine).toBeDefined();
    engine.destroy();
  });

  test('should create conversation', () => {
    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    const conversation = engine.createConversation();
    expect(conversation).toBeDefined();
    conversation.destroy();
    engine.destroy();
  });

  test('should create conversation with system instruction', () => {
    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    const conversation = engine.createConversationWithSystem('You are a helpful assistant.');
    expect(conversation).toBeDefined();
    conversation.destroy();
    engine.destroy();
  });

  test('should send message and receive response', () => {
    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    const conversation = engine.createConversation();

    const response = conversation.sendMessage('user', 'Hello');
    expect(response).toBeDefined();
    expect(typeof response).toBe('string');
    expect(response.length).toBeGreaterThan(0);

    conversation.destroy();
    engine.destroy();
  });

  test('should get benchmark info if enabled', () => {
    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    const conversation = engine.createConversation();

    conversation.sendMessage('user', 'Test');
    const benchmark = conversation.getBenchmarkInfo();

    // Benchmark may be null if not enabled in the model
    if (benchmark) {
      expect(benchmark.prefillTurns).toBeDefined();
      expect(benchmark.decodeTurns).toBeDefined();
      expect(typeof benchmark.timeToFirstTokenMs).toBe('number');
    }

    conversation.destroy();
    engine.destroy();
  });

  test('should throw error when using destroyed engine', () => {
    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    engine.destroy();

    expect(() => {
      engine.createConversation();
    }).toThrow('Engine has been destroyed');
  });

  test('should throw error when using destroyed conversation', () => {
    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    const conversation = engine.createConversation();
    conversation.destroy();

    expect(() => {
      conversation.sendMessage('user', 'Hello');
    }).toThrow('Conversation has been destroyed');

    engine.destroy();
  });

  test('should create conversation with tools', () => {
    const tools: ToolDefinition[] = [
      {
        name: 'get_weather',
        description: 'Get weather for a location',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string', description: 'City name' },
          },
          required: ['location'],
        },
      },
    ];

    const engine = new LiteRTEngine(TEST_MODEL_PATH, LiteRTBackend.CPU);
    const conversation = engine.createConversationWithTools(
      tools,
      'You are a helpful assistant.',
    );
    expect(conversation).toBeDefined();
    conversation.destroy();
    engine.destroy();
  });
});

describe('parseToolCalls', () => {
  test('should parse Gemma-style function calls with <escape> tags', () => {
    const text =
      '<start_function_call>call:get_weather{location:<escape>Paris<escape>}<end_function_call>';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('get_weather');
    expect(calls[0].arguments.location).toBe('Paris');
  });

  test('should parse function calls with multiple parameters', () => {
    const text =
      '<start_function_call>call:send_message{recipient:<escape>John<escape>,message:<escape>Hello there<escape>}<end_function_call>';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('send_message');
    expect(calls[0].arguments.recipient).toBe('John');
    expect(calls[0].arguments.message).toBe('Hello there');
  });

  test('should parse simple key:value pairs without escape tags', () => {
    const text =
      '<start_function_call>call:set_alarm{time:07:30,label:Morning}<end_function_call>';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('set_alarm');
    expect(calls[0].arguments.time).toBe('07:30');
    expect(calls[0].arguments.label).toBe('Morning');
  });

  test('should parse multiple function calls in one response', () => {
    const text =
      '<start_function_call>call:get_weather{location:<escape>Paris<escape>}<end_function_call> ' +
      '<start_function_call>call:get_weather{location:<escape>London<escape>}<end_function_call>';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(2);
    expect(calls[0].arguments.location).toBe('Paris');
    expect(calls[1].arguments.location).toBe('London');
  });

  test('should parse JSON tool_calls format', () => {
    const text = JSON.stringify({
      tool_calls: [
        {
          type: 'function',
          function: {
            name: 'get_weather',
            arguments: { location: 'Tokyo' },
          },
        },
      ],
    });
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('get_weather');
    expect(calls[0].arguments.location).toBe('Tokyo');
  });

  test('should return empty array for plain text', () => {
    const text = 'The weather in Paris is sunny today.';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(0);
  });

  test('should handle empty function arguments', () => {
    const text =
      '<start_function_call>call:list_alarms{}<end_function_call>';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('list_alarms');
    expect(Object.keys(calls[0].arguments)).toHaveLength(0);
  });
});
