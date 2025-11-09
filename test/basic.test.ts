/**
 * Basic tests for LiteRT-LM Node.js bindings
 */

import { test, expect, describe } from 'bun:test';
import { LiteRTEngine, LiteRTBackend, LiteRTConversation } from '../src/index';

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
});
