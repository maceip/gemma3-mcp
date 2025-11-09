#!/usr/bin/env bun
/**
 * Simple test of LiteRT-LM Node.js bindings
 */

import { LiteRTEngine, LiteRTBackend } from '../dist/index.js';

async function main() {
  const modelPath = process.argv[2] || '/Users/rpm/Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.tflite';

  console.log('🚀 Initializing LiteRT-LM Engine...');
  console.log(`   Model: ${modelPath}`);
  console.log(`   Backend: GPU (with CPU fallback)`);

  try {
    // Create engine
    const engine = new LiteRTEngine(modelPath, LiteRTBackend.GPU);
    console.log(`✅ Engine created with ${engine.getBackend()} backend`);

    // Create conversation
    const conversation = engine.createConversationWithSystem(
      'You are a helpful AI assistant.'
    );
    console.log('✅ Conversation created\n');

    // Send a test message
    console.log('User: Hello! What can you help me with?');
    const response = conversation.sendMessage('user', 'Hello! What can you help me with?');
    console.log(`\nAssistant: ${response}\n`);

    // Get benchmark info
    const benchmark = conversation.getBenchmarkInfo();
    if (benchmark) {
      console.log('📊 Benchmark Info:');
      console.log(`   Time to first token: ${benchmark.timeToFirstTokenMs.toFixed(2)}ms`);

      if (benchmark.decodeTurns.length > 0) {
        const lastDecode = benchmark.decodeTurns[benchmark.decodeTurns.length - 1];
        console.log(`   Tokens generated: ${lastDecode.numTokens}`);
        console.log(`   Tokens/sec: ${lastDecode.tokensPerSec.toFixed(2)}`);
        console.log(`   Duration: ${(lastDecode.durationSeconds * 1000).toFixed(0)}ms`);
      }
    }

    // Cleanup
    conversation.destroy();
    engine.destroy();
    console.log('\n✅ Test completed successfully!');

  } catch (error) {
    console.error(`❌ Error: ${error}`);
    process.exit(1);
  }
}

main().catch(console.error);
