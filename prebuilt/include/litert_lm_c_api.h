#ifndef LITERT_LM_RUST_API_H_
#define LITERT_LM_RUST_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointers for C API
typedef void* LiteRtLmEnginePtr;
typedef void* LiteRtLmConversationPtr;

// Backend types
typedef enum {
  LITERT_LM_BACKEND_CPU = 0,
  LITERT_LM_BACKEND_GPU = 1,
} LiteRtLmBackend;

// Status codes
typedef enum {
  LITERT_LM_OK = 0,
  LITERT_LM_ERROR = -1,
  LITERT_LM_ERROR_INVALID_ARGS = -2,
  LITERT_LM_ERROR_NOT_INITIALIZED = -3,
  LITERT_LM_ERROR_MODEL_LOAD_FAILED = -4,
  LITERT_LM_ERROR_GENERATION_FAILED = -5,
} LiteRtLmStatus;

// ============================================================================
// Engine API
// ============================================================================

/**
 * Create a new LiteRT-LM Engine.
 *
 * @param model_path Path to the .litertlm model file
 * @param backend Backend to use (CPU or GPU)
 * @param out_engine Output pointer for the created engine
 * @return Status code (0 = success, negative = error)
 */
int LiteRtLmEngine_Create(
    const char* model_path,
    LiteRtLmBackend backend,
    LiteRtLmEnginePtr* out_engine);

/**
 * Destroy an engine and free resources.
 *
 * @param engine Engine to destroy
 */
void LiteRtLmEngine_Destroy(LiteRtLmEnginePtr engine);

// ============================================================================
// Conversation API
// ============================================================================

/**
 * Create a new Conversation with default config.
 *
 * @param engine Engine to use
 * @param out_conversation Output pointer for the created conversation
 * @return Status code (0 = success, negative = error)
 */
int LiteRtLmConversation_Create(
    LiteRtLmEnginePtr engine,
    LiteRtLmConversationPtr* out_conversation);

/**
 * Create a new Conversation with system instruction.
 *
 * @param engine Engine to use
 * @param system_instruction System instruction for the conversation (can be NULL)
 * @param out_conversation Output pointer for the created conversation
 * @return Status code (0 = success, negative = error)
 */
int LiteRtLmConversation_CreateWithSystem(
    LiteRtLmEnginePtr engine,
    const char* system_instruction,
    LiteRtLmConversationPtr* out_conversation);

/**
 * Create a new Conversation with tools and system instruction.
 *
 * @param engine Engine to use
 * @param system_instruction System instruction for the conversation (can be NULL)
 * @param tools_json JSON array of tool declarations (can be NULL)
 * @param out_conversation Output pointer for the created conversation
 * @return Status code (0 = success, negative = error)
 */
int LiteRtLmConversation_CreateWithTools(
    LiteRtLmEnginePtr engine,
    const char* system_instruction,
    const char* tools_json,
    LiteRtLmConversationPtr* out_conversation);

/**
 * Send a message to the conversation (blocking).
 *
 * @param conversation Conversation instance
 * @param role Message role ("user", "model", "system")
 * @param content Message content (text)
 * @param out_response Output pointer for the response text (must be freed with LiteRtLm_FreeString)
 * @return Status code (0 = success, negative = error)
 */
int LiteRtLmConversation_SendMessage(
    LiteRtLmConversationPtr conversation,
    const char* role,
    const char* content,
    char** out_response);

/**
 * Destroy a conversation and free resources.
 *
 * @param conversation Conversation to destroy
 */
void LiteRtLmConversation_Destroy(LiteRtLmConversationPtr conversation);

// ============================================================================
// Benchmark API
// ============================================================================

/**
 * Benchmark data for a single turn (prefill or decode).
 */
typedef struct {
  uint64_t num_tokens;       // Number of tokens processed
  double duration_seconds;   // Time taken in seconds
  double tokens_per_sec;     // Throughput (tokens/second)
} LiteRtLmTurnBenchmark;

/**
 * Full benchmark information for a conversation.
 */
typedef struct {
  // Prefill metrics
  uint32_t total_prefill_turns;
  LiteRtLmTurnBenchmark* prefill_turns;  // Array of length total_prefill_turns

  // Decode metrics
  uint32_t total_decode_turns;
  LiteRtLmTurnBenchmark* decode_turns;   // Array of length total_decode_turns

  // Calculated metrics
  double time_to_first_token_ms;

  // Last turn shortcuts (for convenience)
  uint64_t last_prefill_token_count;
  uint64_t last_decode_token_count;
} LiteRtLmBenchmarkInfo;

/**
 * Get benchmark information for a conversation.
 *
 * @param conversation Conversation instance
 * @param out_benchmark Output pointer for benchmark info (must be freed with LiteRtLm_FreeBenchmark)
 * @return Status code (0 = success, -1 = benchmark not enabled, other = error)
 */
int LiteRtLmConversation_GetBenchmarkInfo(
    LiteRtLmConversationPtr conversation,
    LiteRtLmBenchmarkInfo** out_benchmark);

/**
 * Free a benchmark info structure allocated by the library.
 *
 * @param benchmark Benchmark info to free
 */
void LiteRtLm_FreeBenchmark(LiteRtLmBenchmarkInfo* benchmark);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Free a string allocated by the library.
 *
 * @param str String to free
 */
void LiteRtLm_FreeString(char* str);

/**
 * Get the last error message.
 *
 * @return Error message string (do not free)
 */
const char* LiteRtLm_GetLastError();

#ifdef __cplusplus
}
#endif

#endif  // LITERT_LM_RUST_API_H_
