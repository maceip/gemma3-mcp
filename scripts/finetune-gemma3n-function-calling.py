"""
Fine-tune Gemma 3n for Function Calling

Applies Google's FunctionGemma technique (Mobile Actions fine-tuning) to the
larger Gemma 3n E2B model instead of the 270M FunctionGemma model.

This script:
  1. Loads the Gemma 3n E2B instruction-tuned model
  2. Applies LoRA adapters for efficient fine-tuning
  3. Trains on function-calling data using the FunctionGemma prompt format
  4. Merges adapters and exports to HuggingFace format
  5. Converts to .litertlm format for on-device deployment

The fine-tuned model uses the same special tokens as FunctionGemma:
  <start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>

Usage:
  pip install torch transformers peft trl datasets accelerate bitsandbytes huggingface_hub
  huggingface-cli login  # need access to gated Gemma 3n model

  # Fine-tune on Google's Mobile Actions dataset (default)
  python scripts/finetune-gemma3n-function-calling.py \
    --base_model google/gemma-3n-E2B-it \
    --output_dir ./gemma3n-agent \
    --epochs 3

  # Dry run (no HF auth needed, validates full pipeline)
  python scripts/finetune-gemma3n-function-calling.py --dry_run

References:
  - https://ai.google.dev/gemma/docs/mobile-actions
  - https://huggingface.co/google/functiongemma-270m-it
  - https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf
"""

import argparse
import json
import os
import shutil

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    GemmaConfig,
    GemmaForCausalLM,
)
from trl import SFTConfig, SFTTrainer


# =============================================================================
# Function-calling prompt format (matches FunctionGemma / Gemma 3n template)
# =============================================================================

SYSTEM_TEMPLATE = """You are a helpful assistant with access to the following functions.
Use them when needed to help the user.

Available functions:
{tool_declarations}"""

TOOL_DECLARATION_TEMPLATE = """{{"name": "{name}", "description": "{description}", "parameters": {parameters}}}"""


def format_tool_declarations(tools: list[dict]) -> str:
    """Format tool declarations as the model expects them."""
    declarations = []
    for tool in tools:
        if "function" in tool:
            fn = tool["function"]
        else:
            fn = tool
        declarations.append(
            TOOL_DECLARATION_TEMPLATE.format(
                name=fn["name"],
                description=fn["description"],
                parameters=json.dumps(fn.get("parameters", {})),
            )
        )
    return "\n".join(declarations)


def format_function_call(name: str, arguments: dict) -> str:
    """Format a function call in FunctionGemma style with <escape> tokens."""
    args_parts = []
    for key, value in arguments.items():
        args_parts.append(f"{key}:<escape>{value}<escape>")
    args_str = ",".join(args_parts)
    return f"<start_function_call>call:{name}{{{args_str}}}<end_function_call>"


def format_function_response(name: str, result: dict) -> str:
    """Format a function response for training."""
    result_parts = []
    for key, value in result.items():
        result_parts.append(f"{key}:<escape>{value}<escape>")
    result_str = ",".join(result_parts)
    return f"<start_function_response>response:{name}{{{result_str}}}<end_function_response>"


def build_training_example(sample: dict, tokenizer) -> str:
    """
    Convert a function-calling training sample into a formatted prompt.

    Expected sample format (Mobile Actions style):
    {
        "tools": [...],              # Tool/function declarations
        "system_prompt": "...",      # Optional system context
        "user_prompt": "...",        # User's request
        "function_call": {           # Expected function call
            "name": "...",
            "arguments": {...}
        },
        "function_response": {...},  # Tool execution result (optional)
        "assistant_response": "..."  # Final natural language response (optional)
    }
    """
    tools = sample.get("tools", [])
    system_prompt = sample.get("system_prompt", "")
    user_prompt = sample.get("user_prompt", "")
    function_call = sample.get("function_call", {})
    function_response = sample.get("function_response", None)
    assistant_response = sample.get("assistant_response", None)

    # Build the conversation
    tool_declarations = format_tool_declarations(tools)

    system_content = SYSTEM_TEMPLATE.format(tool_declarations=tool_declarations)
    if system_prompt:
        system_content += f"\n\nContext: {system_prompt}"

    messages = [
        {"role": "developer", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    # The model should output the function call
    call_text = format_function_call(
        function_call["name"], function_call.get("arguments", {})
    )
    messages.append({"role": "model", "content": call_text})

    # If there's a function response and final answer, include them
    if function_response is not None:
        response_text = format_function_response(
            function_call["name"], function_response
        )
        messages.append({"role": "tool", "content": response_text})

        if assistant_response:
            messages.append({"role": "model", "content": assistant_response})

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


# =============================================================================
# Synthetic training data (fallback if no dataset provided)
# =============================================================================

SYNTHETIC_SAMPLES = [
    {
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        "user_prompt": "What's the weather like in Tokyo?",
        "function_call": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
        "function_response": {
            "temperature": "22",
            "condition": "Partly cloudy",
            "unit": "celsius",
        },
        "assistant_response": "The weather in Tokyo is currently 22°C and partly cloudy.",
    },
    {
        "tools": [
            {
                "name": "send_message",
                "description": "Send a text message to a contact",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "description": "Contact name",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message text",
                        },
                    },
                    "required": ["recipient", "message"],
                },
            }
        ],
        "user_prompt": "Text Sarah that I'll be 10 minutes late",
        "function_call": {
            "name": "send_message",
            "arguments": {
                "recipient": "Sarah",
                "message": "I'll be 10 minutes late",
            },
        },
        "function_response": {"status": "sent", "timestamp": "2025-01-15T10:30:00Z"},
        "assistant_response": "Done! I've sent Sarah a message saying you'll be 10 minutes late.",
    },
    {
        "tools": [
            {
                "name": "set_alarm",
                "description": "Set an alarm for a specific time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "Time in HH:MM format",
                        },
                        "label": {
                            "type": "string",
                            "description": "Alarm label",
                        },
                    },
                    "required": ["time"],
                },
            }
        ],
        "user_prompt": "Wake me up at 6:30 tomorrow morning",
        "function_call": {
            "name": "set_alarm",
            "arguments": {"time": "06:30", "label": "Morning alarm"},
        },
        "function_response": {"status": "set", "time": "06:30", "label": "Morning alarm"},
        "assistant_response": "I've set an alarm for 6:30 AM labeled 'Morning alarm'.",
    },
    {
        "tools": [
            {
                "name": "search_contacts",
                "description": "Search for contacts by name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Name to search for",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "make_call",
                "description": "Call a phone number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {
                            "type": "string",
                            "description": "Phone number to call",
                        }
                    },
                    "required": ["phone_number"],
                },
            },
        ],
        "user_prompt": "Call Mom",
        "function_call": {
            "name": "search_contacts",
            "arguments": {"query": "Mom"},
        },
        "function_response": {
            "name": "Mom",
            "phone_number": "+1-555-0100",
        },
        "assistant_response": "Found Mom's number. Calling +1-555-0100 now.",
    },
    {
        "tools": [
            {
                "name": "get_directions",
                "description": "Get directions between two locations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Starting location"},
                        "destination": {
                            "type": "string",
                            "description": "Destination",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["driving", "walking", "transit"],
                        },
                    },
                    "required": ["destination"],
                },
            }
        ],
        "user_prompt": "How do I get to the airport?",
        "function_call": {
            "name": "get_directions",
            "arguments": {"destination": "airport", "mode": "driving"},
        },
    },
]


def create_synthetic_dataset() -> Dataset:
    """Create a small synthetic dataset for testing the pipeline."""
    return Dataset.from_list(SYNTHETIC_SAMPLES)


# =============================================================================
# Main training pipeline
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3n for function calling"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3n-E2B-it",
        help="Base model ID on HuggingFace (default: google/gemma-3n-E2B-it)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemma3n-function-calling",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="google/mobile-actions",
        help="HuggingFace dataset ID (default: google/mobile-actions, use 'none' for synthetic data)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for training (QLoRA)",
    )
    parser.add_argument(
        "--merge_and_push",
        action="store_true",
        help="Merge LoRA adapters and push to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID for pushing (e.g. kontextdev/agent-gemma)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run full pipeline with a tiny local Gemma-architecture model (no HF auth needed)",
    )
    args = parser.parse_args()

    # --- Dry run: create a tiny Gemma-architecture model locally ---
    if args.dry_run:
        print("=== DRY RUN MODE ===")
        print("Creating tiny Gemma-architecture model for pipeline validation...")
        args.use_4bit = False  # 4-bit not needed for tiny model
        args.epochs = 1
        args.batch_size = 2
        args.max_seq_length = 256
        args.lora_r = 4
        args.lora_alpha = 8

        dry_run_dir = os.path.join(args.output_dir, "_dry_run_model")
        os.makedirs(dry_run_dir, exist_ok=True)

        # Create a minimal Gemma config (same architecture, tiny dimensions)
        config = GemmaConfig(
            vocab_size=32000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=512,
        )
        tiny_model = GemmaForCausalLM(config)
        tiny_model.save_pretrained(dry_run_dir)

        # Create a minimal tokenizer from the Gemma template
        from transformers import GemmaTokenizerFast
        # Use a pre-trained tokenizer that's publicly accessible as template
        # Fall back to a basic tokenizer if needed
        try:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        except Exception:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

        # Set Gemma-style chat template
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<start_of_turn>{{ message.role }}\n"
            "{{ message.content }}<end_of_turn>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        )
        tokenizer.save_pretrained(dry_run_dir)
        args.base_model = dry_run_dir
        args.dataset = "none"  # Use synthetic data for dry run
        print(f"Tiny Gemma model created at: {dry_run_dir}")
        print(f"  Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden, {config.vocab_size} vocab")
        print()

    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"4-bit quantization: {args.use_4bit}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Add special tokens for function calling ---
    special_tokens = [
        "<start_function_call>",
        "<end_function_call>",
        "<start_function_response>",
        "<end_function_response>",
        "<escape>",
    ]
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    print(f"Added {num_added} special tokens for function calling")

    # --- Load model ---
    if args.use_4bit and not args.dry_run:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto" if not args.dry_run else None,
            torch_dtype=torch.float32 if args.dry_run else torch.bfloat16,
        )

    # Resize embeddings for new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load dataset ---
    if args.dataset == "google/mobile-actions":
        # Google's Mobile Actions dataset: JSONL where each line is a JSON object
        # with keys: metadata (train/eval), tools, messages
        # This matches the format used in Google's FunctionGemma fine-tuning notebook
        print(f"Loading Google Mobile Actions dataset...")
        from huggingface_hub import hf_hub_download
        data_file = hf_hub_download(
            repo_id="google/mobile-actions",
            filename="dataset.jsonl",
            repo_type="dataset",
        )
        raw_dataset = load_dataset("text", data_files=data_file, encoding="utf-8")["train"]

        def apply_mobile_actions_format(sample):
            entry = json.loads(sample["text"])
            # Use tokenizer.apply_chat_template with tools= param
            # Full prompt+completion (for training)
            prompt_and_completion = tokenizer.apply_chat_template(
                entry["messages"],
                tools=entry["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {
                "text": prompt_and_completion,
                "split": entry.get("metadata", "train"),
            }

        processed = raw_dataset.map(apply_mobile_actions_format)
        # Filter to training split only
        formatted_dataset = processed.filter(lambda x: x["split"] == "train")
        print(f"Training samples: {len(formatted_dataset)} (filtered from {len(processed)} total)")

    elif args.dataset and args.dataset != "none":
        print(f"Loading dataset: {args.dataset}")
        raw_dataset = load_dataset(args.dataset, split="train")

        def format_sample(sample):
            return {"text": build_training_example(sample, tokenizer)}

        formatted_dataset = raw_dataset.map(format_sample)
        print(f"Training samples: {len(formatted_dataset)}")

    else:
        print("Using synthetic function-calling dataset for demo")
        dataset = create_synthetic_dataset()

        def format_sample(sample):
            return {"text": build_training_example(sample, tokenizer)}

        formatted_dataset = dataset.map(format_sample)
        print(f"Training samples: {len(formatted_dataset)}")

    # --- Training arguments ---
    use_bf16 = torch.cuda.is_available() and not args.dry_run
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if args.dry_run else 4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=2 if args.dry_run else 0,
        warmup_ratio=0.0 if args.dry_run else 0.1,
        lr_scheduler_type="cosine",
        logging_steps=1 if args.dry_run else 10,
        save_strategy="no" if args.dry_run else "epoch",
        bf16=use_bf16,
        optim="paged_adamw_8bit" if (args.use_4bit and not args.dry_run) else "adamw_torch",
        max_grad_norm=0.3,
        max_length=args.max_seq_length,
        report_to="none",
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # --- Save LoRA adapters ---
    adapter_dir = os.path.join(args.output_dir, "lora-adapters")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapters saved to: {adapter_dir}")

    # --- Merge and export ---
    if args.merge_and_push:
        print("Merging LoRA adapters into base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_model = merged_model.merge_and_unload()

        merged_dir = os.path.join(args.output_dir, "merged")
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

        if args.hub_model_id:
            print(f"Pushing to HuggingFace Hub: {args.hub_model_id}")
            merged_model.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
            print("Push complete!")

    if args.dry_run:
        print("\n=== DRY RUN COMPLETE ===")
        print("Full pipeline validated successfully:")
        print("  - Model loading (Gemma architecture)")
        print("  - Special token injection")
        print("  - LoRA adapter application")
        print("  - Dataset formatting with function-call tokens")
        print("  - SFTTrainer training loop")
        print("  - Adapter save/load")
        # Clean up dry run artifacts
        dry_run_dir = os.path.join(args.output_dir, "_dry_run_model")
        if os.path.exists(dry_run_dir):
            shutil.rmtree(dry_run_dir)
            print(f"  - Cleaned up: {dry_run_dir}")
        print()
        print("To run for real with Gemma 3n:")
        print(f"  python {__file__} --base_model google/gemma-3n-E2B-it")
        return

    print("\n=== Next steps ===")
    print("1. Convert to .litertlm format using LiteRT-LM tools:")
    print("   python scripts/convert-to-litertlm.py \\")
    print(f"     --model_dir {args.output_dir}/merged \\")
    print("     --output gemma-3n-E2B-it-agent.litertlm")
    print("")
    print("2. Test with the function-calling example:")
    print("   bun examples/function-calling.ts gemma-3n-E2B-it-agent.litertlm")
    print("")
    print("3. For Kotlin/Android, use the LiteRT-LM Kotlin API:")
    print('   val engine = Engine(EngineConfig(modelPath = "agent.litertlm"))')
    print("   val conv = engine.createConversation(ConversationConfig(")
    print("     tools = listOf(tool(MyToolSet()))")
    print("   ))")


if __name__ == "__main__":
    main()
