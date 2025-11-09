# tres vibes: accelerated model serving over mcp

Native Node.js bindings for [LiteRT-LM](https://github.com/google/litert-lm) with an API matching the `mcp-llm` Rust crate from [assist-mcp](~/assist-mcp).

## Features

- **Native Performance**: Direct bindings to LiteRT-LM C++ library via node-gyp
- **TypeScript Support**: Full TypeScript types and IDE autocomplete
- **mcp-llm Compatible API**: Same interface as the Rust `mcp-llm` crate
- **Conversation Management**: Multi-turn conversations with context
- **Benchmarking**: Built-in performance metrics (tokens/sec, latency)
- **CPU & GPU**: Support for both CPU and GPU backends

## Prerequisites

1. **Prepare LiteRT-LM runtime artifacts**  
   This repository expects prebuilt LiteRT-LM headers and shared libraries inside `prebuilt/<platform>/`.

   To build them yourself from the LiteRT-LM sources, run:

   ```bash
   cd ~/LiteRT-LM
   bazel build //rust_api:litert_lm_rust_api \
     --define=litert_link_capi_so=true \
     --define=resolve_symbols_in_exec=false
   ```

   Copy the resulting headers and shared libraries into this repo:

   ```bash
   # Headers
   mkdir -p /Users/rpm/gemma3-mcp/prebuilt/include
   cp rust_api/litert_lm_c_api.h /Users/rpm/gemma3-mcp/prebuilt/include/

   # macOS example (adjust for linux/windows)
   mkdir -p /Users/rpm/gemma3-mcp/prebuilt/darwin/lib
   cp bazel-bin/rust_api/*.dylib /Users/rpm/gemma3-mcp/prebuilt/darwin/lib/
   ```

   The build system will automatically copy these `.dylib/.so/.dll` files next to the compiled addon so that they can be redistributed without a full LiteRT-LM checkout.

2. **Install dependencies**:
   ```bash
   bun install
   ```

## Installation

### Quick Start with Gemini

Install directly as a Gemini extension:

```bash
gemini extensions install https://github.com/maceip/tres-gemma
```

Once installed, the extension provides on-device LLM acceleration through the Gemini CLI.

### Manual Installation

```bash
# Install dependencies (verifies & copies prebuilt runtime)
bun install

# Build native addon and TypeScript
bun run build
```

## Usage

### As a Gemini Extension

After installing via `gemini extensions install`, the extension is automatically available in Gemini:

```bash
# The extension runs automatically with Gemini commands
# Provides accelerated on-device model inference
gemini chat "Hello, how are you?"

# Check extension status
gemini extensions list
```

### Basic Example

```typescript
import { LiteRTEngine, LiteRTBackend } from 'litert-lm-node';

// Create engine
const engine = new LiteRTEngine(
  '/path/to/model.litertlm',
  LiteRTBackend.GPU // defaults to GPU with automatic CPU fallback
);

// Create conversation
const conversation = engine.createConversationWithSystem(
  'You are a helpful AI assistant.'
);

// Send messages
const response = conversation.sendMessage('user', 'Hello!');
console.log('Assistant:', response);

// Get performance metrics
const benchmark = conversation.getBenchmarkInfo();
if (benchmark) {
  console.log('Tokens/sec:', benchmark.decodeTurns[0].tokensPerSec);
}

// Cleanup
conversation.destroy();
engine.destroy();
```

### Interactive Chat

```bash
bun examples/chat.ts /path/to/model.litertlm
```

## API Reference

### `LiteRTEngine`

Main engine class for loading models.

```typescript
class LiteRTEngine {
  constructor(modelPath: string, backend?: LiteRTBackend);
  createConversation(): LiteRTConversation;
  createConversationWithSystem(systemInstruction: string): LiteRTConversation;
  destroy(): void;
}
```

### `LiteRTConversation`

Manages a conversation session.

```typescript
class LiteRTConversation {
  sendMessage(role: MessageRole, content: string): string;
  getBenchmarkInfo(): BenchmarkInfo | null;
  destroy(): void;
}
```

### Types

```typescript
enum LiteRTBackend {
  CPU = 'cpu',
  GPU = 'gpu',
}

type MessageRole = 'user' | 'model' | 'system';

interface BenchmarkInfo {
  prefillTurns: TurnBenchmark[];
  decodeTurns: TurnBenchmark[];
  timeToFirstTokenMs: number;
  lastPrefillTokenCount: number;
  lastDecodeTokenCount: number;
}

interface TurnBenchmark {
  numTokens: number;
  durationSeconds: number;
  tokensPerSec: number;
}
```

## Matching mcp-llm API

This library provides the same interface as the Rust `mcp-llm` crate:

| mcp-llm (Rust) | litert-lm-node (TypeScript) |
|----------------|----------------------------|
| `LiteRTEngine::new()` | `new LiteRTEngine()` |
| `engine.create_conversation()` | `engine.createConversation()` |
| `conversation.send_message()` | `conversation.sendMessage()` |
| `conversation.get_benchmark_info()` | `conversation.getBenchmarkInfo()` |
| `LiteRTBackend::Cpu` | `LiteRTBackend.CPU` |

## Testing

```bash
# Set test model path
export LITERT_TEST_MODEL=/path/to/model.litertlm

# Run tests
bun test
```

## Building

```bash
# Clean build
bun run clean

# Rebuild addon + TypeScript (handles prebuilt runtime copy)
bun run build

# Build TypeScript only
bun run build:ts
```

## Architecture

```
┌─────────────────────────────────────┐
│   TypeScript Application Layer       │
├─────────────────────────────────────┤
│   src/index.ts (Type-safe wrapper)   │
├─────────────────────────────────────┤
│   src/native/litert_addon.cc         │
│   (N-API C++ bindings)               │
├─────────────────────────────────────┤
│   LiteRT-LM Rust API                 │
│   (~/LiteRT-LM/rust_api)             │
├─────────────────────────────────────┤
│   LiteRT-LM C++ Engine               │
│   (~/LiteRT-LM/c/engine.cc)          │
└─────────────────────────────────────┘
```

## License

MIT

## Related Projects

- [LiteRT-LM](https://github.com/google/litert-lm) - On-device language models
- [assist-mcp](~/assist-mcp) - MCP server with Rust bindings

## Hackathon

This Gemini extension exists to make it effortless to run high-performance, on-device machine learning models. It packages the LiteRT-LM runtime, GPU-first bindings, and MCP-compatible tooling into a single, hackathon-friendly bundle so teams can focus on building experiences rather than wiring up infrastructure.

## Video

Clip 1 — 0-6s: “Maker uploads a fresh prototype to Gemini on a laptop; status flashes ‘On-device acceleration ready.’ Energetic music cues experimentation.”

Clip 2 — 6-12s: “Gemini instantly renders design highlights with GPU-powered overlays; real-time latency meter stays in the green. Maker smiles at the responsive feedback.”

Clip 3 — 12-18s: “Team huddles around the screen as Gemini simulates multi-user collaboration, syncing devices without the cloud. Fast-paced motion graphics emphasize local speed.”

Clip 4 — 18-24s: “Field test outdoors: a tablet running Gemini classifies objects in milliseconds, overlaying insights. The display shows ‘Powered by on-device LiteRT-LM.’”

Clip 5 — 24-30s: “Final reveal montage: prototype ships, telemetry dashboard charts rising engagement, text overlay reads ‘Gemini + LiteRT-LM: High-performance, on-device ML for everyone.’ Fade out.”
