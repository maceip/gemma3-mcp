"""
Convert a fine-tuned Gemma 3n model to .litertlm format for on-device deployment.

Supports two export paths:
  1. LiteRT-LM native conversion (recommended for production)
  2. optimum-executorch (.pte format, alternative for ExecuTorch-based apps)

The .litertlm format bundles:
  - INT4-quantized TFLite model weights
  - SentencePiece tokenizer
  - LLM metadata including the Jinja chat template
  - Vision/audio encoders (if multimodal)

Usage:
  # Path 1: LiteRT-LM format (requires LiteRT-LM build tools)
  python scripts/convert-to-litertlm.py \
    --model_dir ./gemma3n-function-calling/merged \
    --output gemma-3n-E2B-it-agent.litertlm \
    --quantize int4

  # Path 2: ExecuTorch format via optimum-executorch
  python scripts/convert-to-litertlm.py \
    --model_dir ./gemma3n-function-calling/merged \
    --output gemma-3n-E2B-it-agent.pte \
    --format executorch \
    --quantize 8da4w

References:
  - https://github.com/google-ai-edge/LiteRT-LM
  - https://github.com/huggingface/optimum-executorch
  - https://github.com/pytorch/executorch/blob/main/examples/models/gemma3/README.md
"""

import argparse
import json
import os
import subprocess
import sys

# Jinja2 chat template for function-calling Gemma 3n models.
# This template is embedded in the .litertlm metadata section so the
# runtime knows how to format conversations with tool declarations.
FUNCTION_CALLING_JINJA_TEMPLATE = r"""
{%- for message in messages -%}
  {%- if message.role == 'developer' or message.role == 'system' -%}
<start_of_turn>developer
{{ message.content }}
{%- if tools is defined and tools|length > 0 %}

Available tools:
{%- for tool in tools %}
<start_function_declaration>
{%- if tool.function is defined %}
{{ tool.function | tojson }}
{%- else %}
{{ tool | tojson }}
{%- endif %}
<end_function_declaration>
{%- endfor %}
{%- endif %}
<end_of_turn>
  {%- elif message.role == 'user' -%}
<start_of_turn>user
{{ message.content }}<end_of_turn>
  {%- elif message.role == 'model' or message.role == 'assistant' -%}
<start_of_turn>model
{%- if message.tool_calls is defined and message.tool_calls -%}
{%- for tc in message.tool_calls -%}
<start_function_call>call:{{ tc.function.name }}{{ '{' }}
{%- for k, v in tc.function.arguments.items() -%}
{{ k }}:<escape>{{ v }}<escape>
{%- if not loop.last %},{% endif -%}
{%- endfor -%}
{{ '}' }}<end_function_call>
{%- endfor -%}
{%- else -%}
{{ message.content }}
{%- endif -%}
<end_of_turn>
  {%- elif message.role == 'tool' -%}
<start_of_turn>tool
{{ message.content }}<end_of_turn>
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<start_of_turn>model
{%- endif -%}
""".strip()


def convert_litertlm(model_dir: str, output: str, quantize: str):
    """
    Convert to .litertlm format using LiteRT-LM's model packager.

    This requires the LiteRT-LM build tools to be available. The conversion:
    1. Exports the HF model to TFLite format
    2. Applies INT4 quantization
    3. Packages with tokenizer + Jinja template into .litertlm
    """
    print(f"Converting {model_dir} -> {output}")
    print(f"Quantization: {quantize}")

    # Write the Jinja template to a temp file for the packager
    template_path = os.path.join(model_dir, "chat_template.jinja")
    with open(template_path, "w") as f:
        f.write(FUNCTION_CALLING_JINJA_TEMPLATE)
    print(f"Wrote function-calling Jinja template to: {template_path}")

    # Also write as tokenizer_config.json update
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            config = json.load(f)
        config["chat_template"] = FUNCTION_CALLING_JINJA_TEMPLATE
        with open(tokenizer_config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Updated tokenizer_config.json with function-calling chat template")

    # Try LiteRT-LM convert tool
    try:
        cmd = [
            "litert-lm-convert",
            "--model_dir", model_dir,
            "--output", output,
            "--quantize", quantize,
            "--chat_template", template_path,
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Successfully created: {output}")
    except FileNotFoundError:
        print("\nlitert-lm-convert not found. Manual steps:")
        print("1. Build LiteRT-LM from source:")
        print("   git clone https://github.com/google-ai-edge/LiteRT-LM")
        print("   cd LiteRT-LM && bazel build //tools:convert")
        print("")
        print("2. Or use Google AI Edge model conversion:")
        print("   pip install ai-edge-model-converter")
        print("   ai-edge-model-converter \\")
        print(f"     --source_model {model_dir} \\")
        print(f"     --output {output} \\")
        print(f"     --quantize {quantize} \\")
        print(f"     --chat_template {template_path}")
        print("")
        print("3. The Jinja template has been written to:")
        print(f"   {template_path}")
        print("   And embedded in tokenizer_config.json")
        print("")
        print("Make sure to include this template when packaging the model —")
        print("it's what tells LiteRT-LM how to format tool declarations")
        print("and parse <start_function_call>/<end_function_call> tokens.")
        sys.exit(1)


def convert_executorch(model_dir: str, output: str, quantize: str):
    """
    Convert to ExecuTorch .pte format using optimum-executorch.

    This path is for apps that use ExecuTorch (PyTorch on-device) instead
    of LiteRT-LM. React Native ExecuTorch v0.4+ supports tool calling.
    """
    print(f"Converting {model_dir} -> {output} (ExecuTorch .pte)")
    print(f"Quantization: {quantize}")

    try:
        # optimum-cli export
        cmd = [
            "optimum-cli", "export", "executorch",
            "--model", model_dir,
            "--task", "text-generation",
            "--recipe", "xnnpack",
            "--output_dir", os.path.dirname(output) or ".",
        ]
        if quantize:
            cmd.extend(["--qlinear", quantize])

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"ExecuTorch model exported to: {output}")
    except FileNotFoundError:
        print("\noptimum-cli not found. Install optimum-executorch:")
        print("  pip install optimum-executorch@git+https://github.com/huggingface/optimum-executorch.git")
        print("")
        print("Or use the Python API directly:")
        print("  from optimum.executorch import ExecuTorchModelForCausalLM")
        print(f'  model = ExecuTorchModelForCausalLM.from_pretrained("{model_dir}",')
        print('    recipe="xnnpack",')
        print(f'    quantize="{quantize}"')
        print("  )")
        print(f'  model.save_pretrained("{os.path.dirname(output) or "."}")')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert fine-tuned Gemma 3n to on-device format"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to merged HuggingFace model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.litertlm or .pte)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["litertlm", "executorch"],
        default="litertlm",
        help="Output format (default: litertlm)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="int4",
        help="Quantization scheme: int4 (litertlm) or 8da4w/4w (executorch)",
    )
    args = parser.parse_args()

    if args.format == "executorch":
        convert_executorch(args.model_dir, args.output, args.quantize)
    else:
        convert_litertlm(args.model_dir, args.output, args.quantize)


if __name__ == "__main__":
    main()
