/**
 * Kotlin example: Function calling with Gemma 3n on Android via LiteRT-LM.
 *
 * This demonstrates how to use the fine-tuned Gemma 3n agent model with the
 * LiteRT-LM Kotlin API's tool calling support. The customer's Android app
 * uses this pattern to enable on-device function calling.
 *
 * Dependencies (build.gradle.kts):
 *   implementation("com.google.ai.edge.litertlm:litertlm-android:0.9.0-alpha01")
 *
 * The model must be fine-tuned with the FunctionGemma technique applied to
 * Gemma 3n (see scripts/finetune-gemma3n-function-calling.py) and converted
 * to .litertlm format (see scripts/convert-to-litertlm.py).
 */

package com.kontextdev.agentgemma.example

import com.google.ai.edge.litertlm.*

// =============================================================================
// Tool definitions using @Tool annotations (recommended approach)
// =============================================================================

class AgentToolSet : ToolSet {

    @Tool(description = "Get the current weather for a city")
    fun getWeather(
        @ToolParam(description = "City name, e.g. 'San Francisco'") location: String,
        @ToolParam(description = "Temperature unit: celsius or fahrenheit") unit: String = "celsius"
    ): Map<String, Any> {
        // In production, call a real weather API
        return mapOf(
            "location" to location,
            "temperature" to 22,
            "unit" to unit,
            "condition" to "Partly cloudy"
        )
    }

    @Tool(description = "Send a text message to a contact")
    fun sendMessage(
        @ToolParam(description = "Recipient contact name") recipient: String,
        @ToolParam(description = "Message text to send") message: String
    ): Map<String, Any> {
        // In production, send via SMS/messaging API
        return mapOf(
            "status" to "sent",
            "recipient" to recipient,
            "timestamp" to System.currentTimeMillis()
        )
    }

    @Tool(description = "Set an alarm for a specific time")
    fun setAlarm(
        @ToolParam(description = "Time in HH:MM format (24-hour)") time: String,
        @ToolParam(description = "Optional label for the alarm") label: String = "Alarm"
    ): Map<String, Any> {
        // In production, use AlarmManager
        return mapOf(
            "status" to "set",
            "time" to time,
            "label" to label
        )
    }

    @Tool(description = "Search for contacts by name")
    fun searchContacts(
        @ToolParam(description = "Name or partial name to search") query: String
    ): List<Map<String, String>> {
        // In production, query ContactsContract
        return listOf(
            mapOf("name" to "John Smith", "phone" to "+1-555-0123"),
            mapOf("name" to "Jane Smith", "phone" to "+1-555-0456")
        )
    }
}

// =============================================================================
// Alternative: OpenAPI-style tool definition for full control
// =============================================================================

class WeatherOpenApiTool : OpenApiTool {
    override fun getToolDescriptionJsonString(): String = """
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    """.trimIndent()

    override fun execute(paramsJsonString: String): String {
        // Parse paramsJsonString, call weather API, return JSON result
        return """{"temperature": 22, "unit": "celsius", "condition": "Sunny"}"""
    }
}

// =============================================================================
// Main example: Automatic tool calling (recommended)
// =============================================================================

suspend fun automaticToolCallingExample(modelPath: String) {
    println("=== Automatic Tool Calling Example ===")

    // Initialize engine with the fine-tuned Gemma 3n agent model
    val engineConfig = EngineConfig(
        modelPath = modelPath,
        backend = Backend.CPU()  // or Backend.GPU() for faster inference
    )
    val engine = Engine(engineConfig)
    engine.initialize()

    // Create conversation with tools — automatic execution enabled by default
    val conversation = engine.createConversation(
        ConversationConfig(
            systemInstruction = Contents.of(
                "You are a helpful assistant that uses tools to help the user."
            ),
            tools = listOf(
                tool(AgentToolSet())
            ),
            automaticToolCalling = true  // LiteRT-LM executes tools automatically
        )
    )

    // The model will:
    // 1. Generate <start_function_call>call:getWeather{...}<end_function_call>
    // 2. LiteRT-LM intercepts the tool call
    // 3. Calls AgentToolSet.getWeather() with parsed arguments
    // 4. Feeds the result back to the model
    // 5. Returns the final natural language response
    conversation.sendMessageAsync("What's the weather in Tokyo?")
        .collect { message ->
            print(message.toString())
        }

    println()
    conversation.close()
    engine.close()
}

// =============================================================================
// Alternative: Manual tool calling (for custom execution logic)
// =============================================================================

suspend fun manualToolCallingExample(modelPath: String) {
    println("=== Manual Tool Calling Example ===")

    val engineConfig = EngineConfig(
        modelPath = modelPath,
        backend = Backend.CPU()
    )
    val engine = Engine(engineConfig)
    engine.initialize()

    val conversation = engine.createConversation(
        ConversationConfig(
            tools = listOf(
                tool(AgentToolSet())
            ),
            automaticToolCalling = false  // We handle tool execution ourselves
        )
    )

    // Send user message — model may respond with tool calls
    val response = conversation.sendMessage("Set an alarm for 7:30 AM")

    if (response.toolCalls.isNotEmpty()) {
        println("Model requested tool calls:")
        val toolResponses = mutableListOf<Content.ToolResponse>()

        for (toolCall in response.toolCalls) {
            println("  -> ${toolCall.name}(${toolCall.arguments})")

            // Execute the tool (your custom logic here)
            val result = when (toolCall.name) {
                "setAlarm" -> """{"status": "set", "time": "07:30", "label": "Morning"}"""
                else -> """{"error": "Unknown tool"}"""
            }

            toolResponses.add(Content.ToolResponse(toolCall.name, result))
        }

        // Send tool results back to model for final response
        val toolMessage = Message.tool(Contents.of(toolResponses))
        val finalResponse = conversation.sendMessage(toolMessage)
        println("Final response: ${finalResponse.text}")
    } else {
        println("Response: ${response.text}")
    }

    conversation.close()
    engine.close()
}

// =============================================================================
// Alternative: Using OpenAPI tools
// =============================================================================

suspend fun openApiToolExample(modelPath: String) {
    println("=== OpenAPI Tool Example ===")

    val engineConfig = EngineConfig(
        modelPath = modelPath,
        backend = Backend.CPU()
    )
    val engine = Engine(engineConfig)
    engine.initialize()

    // Mix annotated ToolSet and OpenAPI tools
    val conversation = engine.createConversation(
        ConversationConfig(
            tools = listOf(
                tool(WeatherOpenApiTool()),  // OpenAPI-style
                tool(AgentToolSet())         // Annotation-style
            )
        )
    )

    conversation.sendMessageAsync("What's the weather in Paris and set an alarm for 8 AM")
        .collect { print(it.toString()) }

    println()
    conversation.close()
    engine.close()
}

// =============================================================================
// Entry point
// =============================================================================

suspend fun main(args: Array<String>) {
    val modelPath = args.firstOrNull()
        ?: error("Usage: FunctionCallingExample <path/to/gemma-3n-agent.litertlm>")

    automaticToolCallingExample(modelPath)
    println()
    manualToolCallingExample(modelPath)
    println()
    openApiToolExample(modelPath)
}
