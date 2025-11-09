#!/usr/bin/env bun
/**
 * Example chat application using LiteRT-LM Node.js bindings
 *
 * Usage:
 *   bun examples/chat.ts <model_path>
 */

import { LiteRTEngine, LiteRTBackend } from '../src/index';

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.error('Usage: bun examples/chat.ts <model_path>');
    process.exit(1);
  }

  const modelPath = args[0];
  const backend = (args[1] as LiteRTBackend) || LiteRTBackend.CPU;

  console.log('🚀 Initializing LiteRT-LM Engine...');
  console.log(`   Model: ${modelPath}`);
  console.log(`   Backend: ${backend}`);

  // Create engine
  const engine = new LiteRTEngine(modelPath, backend);

  try {
    // Create conversation with system instruction
    const systemInstruction = 'You are a helpful AI assistant.';
    const conversation = engine.createConversationWithSystem(systemInstruction);

    console.log('\n✅ Engine ready! Type your messages (Ctrl+C to exit)\n');

    // Interactive chat loop
    const readline = require('readline');
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: 'You: ',
    });

    rl.prompt();

    rl.on('line', (line: string) => {
      const userMessage = line.trim();

      if (!userMessage) {
        rl.prompt();
        return;
      }

      try {
        // Send user message
        const response = conversation.sendMessage('user', userMessage);

        console.log(`\nAssistant: ${response}\n`);

        // Get benchmark info if available
        const benchmark = conversation.getBenchmarkInfo();
        if (benchmark) {
          const lastDecode = benchmark.decodeTurns[benchmark.decodeTurns.length - 1];
          if (lastDecode) {
            console.log(
              `⚡ ${lastDecode.tokensPerSec.toFixed(2)} tokens/sec | ` +
              `${lastDecode.numTokens} tokens | ` +
              `${(lastDecode.durationSeconds * 1000).toFixed(0)}ms`
            );
          }
          console.log();
        }
      } catch (error) {
        console.error(`Error: ${error}`);
      }

      rl.prompt();
    });

    rl.on('close', () => {
      console.log('\n👋 Goodbye!');
      conversation.destroy();
      engine.destroy();
      process.exit(0);
    });

  } catch (error) {
    console.error(`Failed to initialize: ${error}`);
    engine.destroy();
    process.exit(1);
  }
}

main().catch(console.error);
