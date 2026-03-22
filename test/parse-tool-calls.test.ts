/**
 * Tests for parseToolCalls - pure TypeScript, no native addon needed
 */

import { test, expect, describe } from 'bun:test';
import { parseToolCalls } from '../src/parse-tool-calls';

describe('parseToolCalls - Gemma 3n stock format (tool_code blocks)', () => {
  test('should parse tool_code block with tool_name field', () => {
    const text = '```tool_code\n{"tool_name": "get_weather", "parameters": {"location": "Paris"}}\n```';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('get_weather');
    expect(calls[0].arguments.location).toBe('Paris');
  });

  test('should parse tool_code block with name field', () => {
    const text = '```tool_code\n{"name": "send_message", "parameters": {"recipient": "John", "message": "I\'ll be there in 5 minutes"}}\n```';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('send_message');
    expect(calls[0].arguments.recipient).toBe('John');
  });

  test('should parse multiple tool_code blocks', () => {
    const text = '```tool_code\n{"tool_name": "get_weather", "parameters": {"location": "Paris"}}\n```\nSome text\n```tool_code\n{"tool_name": "get_weather", "parameters": {"location": "London"}}\n```';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(2);
    expect(calls[0].arguments.location).toBe('Paris');
    expect(calls[1].arguments.location).toBe('London');
  });

  test('should convert numeric parameter values to strings', () => {
    const text = '```tool_code\n{"tool_name": "set_timer", "parameters": {"duration": 300}}\n```';
    const calls = parseToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0].arguments.duration).toBe('300');
  });
});

describe('parseToolCalls - FunctionGemma format', () => {
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
