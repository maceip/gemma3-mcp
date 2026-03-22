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

The fine-tuned model uses the same markup as FunctionGemma:
  <start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>

Usage:
  pip install torch transformers peft trl datasets accelerate bitsandbytes huggingface_hub
  huggingface-cli login  # need access to gated Gemma 3n model

  # Fine-tune on Google's Mobile Actions dataset (default)
  python scripts/finetune-gemma3n-function-calling.py \
    --base_model google/gemma-3n-E2B-it \
    --output_dir ./gemma3n-agent \
    --epochs 3

References:
  - https://ai.google.dev/gemma/docs/mobile-actions
  - https://huggingface.co/google/functiongemma-270m-it
  - https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# =============================================================================
# Function-calling chat template for Gemma 3n
# =============================================================================

# The default Gemma 3n tokenizer template enforces strict user/assistant
# alternation and doesn't support developer/tool roles or tool_calls.
# We override it with a template that handles the full function-calling
# conversation flow: developer (system) -> user -> assistant (with tool_calls)
# -> tool (response) -> assistant (final answer).
# This matches how Google's FunctionGemma tokenizer handles the Mobile Actions
# dataset format (developer + user + assistant w/ tool_calls).
FUNCTION_CALLING_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message.role == 'developer' or message.role == 'system' -%}"
    "<start_of_turn>developer\n"
    "{{ message.content }}"
    "{%- if tools is defined and tools|length > 0 %}\n\n"
    "Available tools:"
    "{%- for tool in tools %}\n"
    "<start_function_declaration>"
    "{%- if tool.function is defined %}"
    "{{ tool.function | tojson }}"
    "{%- else %}"
    "{{ tool | tojson }}"
    "{%- endif %}"
    "<end_function_declaration>"
    "{%- endfor %}"
    "{%- endif %}"
    "<end_of_turn>\n"
    "{%- elif message.role == 'user' -%}"
    "<start_of_turn>user\n"
    "{{ message.content }}<end_of_turn>\n"
    "{%- elif message.role == 'model' or message.role == 'assistant' -%}"
    "<start_of_turn>model\n"
    "{%- if message.tool_calls is defined and message.tool_calls -%}"
    "{%- for tc in message.tool_calls -%}"
    "<start_function_call>call:{{ tc.function.name }}{{ '{' }}"
    "{%- for k, v in tc.function.arguments.items() -%}"
    "{{ k }}:<escape>{{ v }}<escape>"
    "{%- if not loop.last %},{% endif -%}"
    "{%- endfor -%}"
    "{{ '}' }}<end_function_call>"
    "{%- endfor -%}"
    "{%- else -%}"
    "{{ message.content }}"
    "{%- endif -%}"
    "<end_of_turn>\n"
    "{%- elif message.role == 'tool' -%}"
    "<start_of_turn>tool\n"
    "{{ message.content }}<end_of_turn>\n"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "<start_of_turn>model\n"
    "{%- endif -%}"
)


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
        help="HuggingFace dataset ID (default: google/mobile-actions)",
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
    args = parser.parse_args()

    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"4-bit quantization: {args.use_4bit}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # NOTE: We do NOT add special tokens for function calling markers like
    # <start_function_call>, <end_function_call>, etc. Gemma 3n is a multimodal
    # model with partitioned embeddings (text, audio, vision). Adding tokens
    # and calling resize_token_embeddings() only resizes the text embedding
    # table, but new token IDs can fall into the audio embedding range (which
    # isn't resized), causing IndexError in embed_audio during forward pass.
    # Instead, the function-calling markers are kept as plain text and get
    # tokenized into subwords. The model learns the pattern during fine-tuning.
    print("Using function-calling markers as plain text (no special token injection)")

    # --- Override chat template for function-calling support ---
    tokenizer.chat_template = FUNCTION_CALLING_CHAT_TEMPLATE
    print("Overrode tokenizer chat template with function-calling template")

    # --- Load model ---
    if args.use_4bit:
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
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

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
        # Messages use roles: developer, user, assistant (with tool_calls)
        print("Loading Google Mobile Actions dataset...")
        from huggingface_hub import hf_hub_download
        data_file = hf_hub_download(
            repo_id="google/mobile-actions",
            filename="dataset.jsonl",
            repo_type="dataset",
        )
        raw_dataset = load_dataset("text", data_files=data_file, encoding="utf-8")["train"]

        def prepare_messages(messages):
            """Ensure tool_calls arguments are dicts (not JSON strings) for the
            Jinja template, which needs to iterate over key-value pairs."""
            prepared = []
            for msg in messages:
                msg = dict(msg)  # shallow copy
                if msg.get("tool_calls"):
                    fixed_calls = []
                    for tc in msg["tool_calls"]:
                        tc = dict(tc)
                        fn = dict(tc.get("function", {}))
                        if isinstance(fn.get("arguments"), str):
                            fn["arguments"] = json.loads(fn["arguments"])
                        tc["function"] = fn
                        fixed_calls.append(tc)
                    msg["tool_calls"] = fixed_calls
                prepared.append(msg)
            return prepared

        def apply_mobile_actions_format(sample):
            entry = json.loads(sample["text"])
            messages = prepare_messages(entry["messages"])
            prompt_and_completion = tokenizer.apply_chat_template(
                messages,
                tools=entry["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {
                "text": prompt_and_completion,
                "split": entry.get("metadata", "train"),
            }

        # --- Validate all samples before training ---
        # Process every sample upfront so we catch template/format errors now,
        # not hours into a training run.
        print("Validating all samples against chat template...")
        errors = []
        for idx in range(len(raw_dataset)):
            try:
                entry = json.loads(raw_dataset[idx]["text"])
                messages = prepare_messages(entry["messages"])
                tokenizer.apply_chat_template(
                    messages,
                    tools=entry["tools"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                errors.append((idx, str(e)))
                if len(errors) <= 5:
                    roles = [m.get("role") for m in entry.get("messages", [])]
                    print(f"  Sample {idx} FAILED: {e}")
                    print(f"    Roles: {roles}")
        if errors:
            print(f"\nERROR: {len(errors)}/{len(raw_dataset)} samples failed template validation.")
            print("Fix the chat template or data before training. Aborting.")
            sys.exit(1)
        print(f"All {len(raw_dataset)} samples passed template validation.")

        processed = raw_dataset.map(apply_mobile_actions_format)
        # Filter to training split only
        formatted_dataset = processed.filter(lambda x: x["split"] == "train")
        print(f"Training samples: {len(formatted_dataset)} (filtered from {len(processed)} total)")

    else:
        print(f"Loading dataset: {args.dataset}")
        raw_dataset = load_dataset(args.dataset, split="train")

        def format_sample(sample):
            """Format using apply_chat_template — expects 'messages' and 'tools' keys."""
            messages = sample.get("messages", [])
            tools = sample.get("tools", [])
            text = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        formatted_dataset = raw_dataset.map(format_sample)
        print(f"Training samples: {len(formatted_dataset)}")

    # --- Training arguments ---
    use_bf16 = torch.cuda.is_available()
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
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
