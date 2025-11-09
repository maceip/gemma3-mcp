/**
 * LiteRT-LM Node.js Native Addon
 *
 * Provides Node.js bindings for LiteRT-LM matching the mcp-llm Rust API
 */

#include <napi.h>
#include <string>
#include <memory>
#include <vector>

// Include LiteRT-LM C API headers
extern "C" {
#include "litert_lm_c_api.h"
}

namespace litert_node {

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Convert LiteRT status code to error message
 */
std::string StatusToError(int status) {
  const char* error_msg = LiteRtLm_GetLastError();
  if (error_msg != nullptr) {
    return std::string(error_msg);
  }

  switch (status) {
    case LITERT_LM_OK:
      return "Success";
    case LITERT_LM_ERROR:
      return "Generic error";
    case LITERT_LM_ERROR_INVALID_ARGS:
      return "Invalid arguments";
    case LITERT_LM_ERROR_NOT_INITIALIZED:
      return "Not initialized";
    case LITERT_LM_ERROR_MODEL_LOAD_FAILED:
      return "Model load failed";
    case LITERT_LM_ERROR_GENERATION_FAILED:
      return "Generation failed";
    default:
      return "Unknown error";
  }
}

/**
 * Throw a JS error from a status code
 */
void ThrowStatusError(Napi::Env env, int status) {
  Napi::Error::New(env, StatusToError(status)).ThrowAsJavaScriptException();
}

// =============================================================================
// LiteRTEngine Class
// =============================================================================

class LiteRTEngine : public Napi::ObjectWrap<LiteRTEngine> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  LiteRTEngine(const Napi::CallbackInfo& info);
  ~LiteRTEngine();

private:
  static Napi::FunctionReference constructor;

  // Methods
  Napi::Value CreateConversation(const Napi::CallbackInfo& info);
  Napi::Value CreateConversationWithSystem(const Napi::CallbackInfo& info);
  Napi::Value Destroy(const Napi::CallbackInfo& info);

  LiteRtLmEnginePtr engine_;
  bool destroyed_;
};

Napi::FunctionReference LiteRTEngine::constructor;

Napi::Object LiteRTEngine::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "LiteRTEngine", {
    InstanceMethod("createConversation", &LiteRTEngine::CreateConversation),
    InstanceMethod("createConversationWithSystem", &LiteRTEngine::CreateConversationWithSystem),
    InstanceMethod("destroy", &LiteRTEngine::Destroy),
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("LiteRTEngine", func);
  return exports;
}

LiteRTEngine::LiteRTEngine(const Napi::CallbackInfo& info)
  : Napi::ObjectWrap<LiteRTEngine>(info), engine_(nullptr), destroyed_(false) {

  Napi::Env env = info.Env();

  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expected 2 arguments: modelPath, backend")
        .ThrowAsJavaScriptException();
    return;
  }

  if (!info[0].IsString() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Arguments must be strings")
        .ThrowAsJavaScriptException();
    return;
  }

  std::string model_path = info[0].As<Napi::String>().Utf8Value();
  std::string backend_str = info[1].As<Napi::String>().Utf8Value();

  // Convert backend string to enum
  LiteRtLmBackend backend;
  if (backend_str == "cpu" || backend_str == "CPU") {
    backend = LITERT_LM_BACKEND_CPU;
  } else if (backend_str == "gpu" || backend_str == "GPU") {
    backend = LITERT_LM_BACKEND_GPU;
  } else {
    Napi::TypeError::New(env, "Backend must be 'cpu' or 'gpu'")
        .ThrowAsJavaScriptException();
    return;
  }

  // Create engine
  int status = LiteRtLmEngine_Create(
      model_path.c_str(),
      backend,
      &engine_
  );

  if (status != LITERT_LM_OK) {
    ThrowStatusError(env, status);
    return;
  }
}

LiteRTEngine::~LiteRTEngine() {
  if (engine_ != nullptr && !destroyed_) {
    LiteRtLmEngine_Destroy(engine_);
    engine_ = nullptr;
    destroyed_ = true;
  }
}

Napi::Value LiteRTEngine::CreateConversation(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (destroyed_ || engine_ == nullptr) {
    Napi::Error::New(env, "Engine has been destroyed")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  LiteRtLmConversationPtr conversation;
  int status = LiteRtLmConversation_Create(engine_, &conversation);

  if (status != LITERT_LM_OK) {
    ThrowStatusError(env, status);
    return env.Null();
  }

  // Create and return conversation wrapper
  Napi::External<void> external = Napi::External<void>::New(env, conversation);
  return external;
}

Napi::Value LiteRTEngine::CreateConversationWithSystem(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (destroyed_ || engine_ == nullptr) {
    Napi::Error::New(env, "Engine has been destroyed")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expected string argument: systemInstruction")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string system_instruction = info[0].As<Napi::String>().Utf8Value();

  LiteRtLmConversationPtr conversation;
  int status = LiteRtLmConversation_CreateWithSystem(
      engine_,
      system_instruction.c_str(),
      &conversation
  );

  if (status != LITERT_LM_OK) {
    ThrowStatusError(env, status);
    return env.Null();
  }

  // Create and return conversation wrapper
  Napi::External<void> external = Napi::External<void>::New(env, conversation);
  return external;
}

Napi::Value LiteRTEngine::Destroy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!destroyed_ && engine_ != nullptr) {
    LiteRtLmEngine_Destroy(engine_);
    engine_ = nullptr;
    destroyed_ = true;
  }

  return env.Undefined();
}

// =============================================================================
// Conversation Helper Functions
// =============================================================================

/**
 * Send message to a conversation
 */
Napi::Value ConversationSendMessage(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3) {
    Napi::TypeError::New(env, "Expected 3 arguments: conversation, role, content")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "First argument must be a conversation handle")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  if (!info[1].IsString() || !info[2].IsString()) {
    Napi::TypeError::New(env, "Role and content must be strings")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  LiteRtLmConversationPtr conversation =
      info[0].As<Napi::External<void>>().Data();

  std::string role = info[1].As<Napi::String>().Utf8Value();
  std::string content = info[2].As<Napi::String>().Utf8Value();

  char* response = nullptr;
  int status = LiteRtLmConversation_SendMessage(
      conversation,
      role.c_str(),
      content.c_str(),
      &response
  );

  if (status != LITERT_LM_OK) {
    ThrowStatusError(env, status);
    return env.Null();
  }

  if (response == nullptr) {
    Napi::Error::New(env, "No response received")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::String result = Napi::String::New(env, response);
  LiteRtLm_FreeString(response);

  return result;
}

/**
 * Destroy a conversation
 */
Napi::Value ConversationDestroy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expected conversation handle")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  LiteRtLmConversationPtr conversation =
      info[0].As<Napi::External<void>>().Data();

  if (conversation != nullptr) {
    LiteRtLmConversation_Destroy(conversation);
  }

  return env.Undefined();
}

/**
 * Get benchmark info from conversation
 */
Napi::Value ConversationGetBenchmarkInfo(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expected conversation handle")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  LiteRtLmConversationPtr conversation =
      info[0].As<Napi::External<void>>().Data();

  LiteRtLmBenchmarkInfo* benchmark = nullptr;
  int status = LiteRtLmConversation_GetBenchmarkInfo(conversation, &benchmark);

  if (status != LITERT_LM_OK) {
    if (status == -1) {
      // Benchmark not enabled
      return env.Null();
    }
    ThrowStatusError(env, status);
    return env.Null();
  }

  if (benchmark == nullptr) {
    return env.Null();
  }

  // Convert benchmark info to JS object
  Napi::Object result = Napi::Object::New(env);

  // Prefill turns
  Napi::Array prefill_turns = Napi::Array::New(env, benchmark->total_prefill_turns);
  for (uint32_t i = 0; i < benchmark->total_prefill_turns; i++) {
    Napi::Object turn = Napi::Object::New(env);
    turn.Set("numTokens", Napi::Number::New(env, benchmark->prefill_turns[i].num_tokens));
    turn.Set("durationSeconds", Napi::Number::New(env, benchmark->prefill_turns[i].duration_seconds));
    turn.Set("tokensPerSec", Napi::Number::New(env, benchmark->prefill_turns[i].tokens_per_sec));
    prefill_turns.Set(i, turn);
  }

  // Decode turns
  Napi::Array decode_turns = Napi::Array::New(env, benchmark->total_decode_turns);
  for (uint32_t i = 0; i < benchmark->total_decode_turns; i++) {
    Napi::Object turn = Napi::Object::New(env);
    turn.Set("numTokens", Napi::Number::New(env, benchmark->decode_turns[i].num_tokens));
    turn.Set("durationSeconds", Napi::Number::New(env, benchmark->decode_turns[i].duration_seconds));
    turn.Set("tokensPerSec", Napi::Number::New(env, benchmark->decode_turns[i].tokens_per_sec));
    decode_turns.Set(i, turn);
  }

  result.Set("prefillTurns", prefill_turns);
  result.Set("decodeTurns", decode_turns);
  result.Set("timeToFirstTokenMs", Napi::Number::New(env, benchmark->time_to_first_token_ms));
  result.Set("lastPrefillTokenCount", Napi::Number::New(env, benchmark->last_prefill_token_count));
  result.Set("lastDecodeTokenCount", Napi::Number::New(env, benchmark->last_decode_token_count));

  LiteRtLm_FreeBenchmark(benchmark);

  return result;
}

// =============================================================================
// Module Initialization
// =============================================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  LiteRTEngine::Init(env, exports);

  // Export conversation helper functions
  exports.Set("conversationSendMessage",
              Napi::Function::New(env, ConversationSendMessage));
  exports.Set("conversationDestroy",
              Napi::Function::New(env, ConversationDestroy));
  exports.Set("conversationGetBenchmarkInfo",
              Napi::Function::New(env, ConversationGetBenchmarkInfo));

  return exports;
}

NODE_API_MODULE(litert_lm_node, Init)

}  // namespace litert_node
