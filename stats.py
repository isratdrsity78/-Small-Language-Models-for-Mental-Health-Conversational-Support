import os
import time
import json
import psutil
import platform
import subprocess
from pathlib import Path

############################################
# CONFIG — EDIT THESE
############################################

MODEL_NAME = "Qwen3-0.6B-finetuned"
MODEL_FORMAT = "GGUF"  # "HF" or "GGUF"
QUANTIZATION = "q4_k_m"  # or "4bit-finetuned", "N/A", etc

MODEL_PATH = "./model.gguf"  # GGUF file OR HF folder
LLAMA_BIN = "./llama.cpp/main"  # path to llama.cpp binary

PROMPT = "Explain why quantization affects model accuracy."
MAX_TOKENS = 128

OUTPUT_JSON = "thesis_metrics.json"

############################################
# SYSTEM INFO
############################################


def get_system_info():
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
    }


############################################
# MODEL SIZE
############################################


def get_model_size_mb(path):
    if os.path.isfile(path):
        return round(os.path.getsize(path) / (1024**2), 2)
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024**2), 2)


############################################
# GGUF INFERENCE TEST
############################################


def run_llama_inference():
    process = psutil.Process()
    ram_before = process.memory_info().rss

    start_time = time.time()

    try:
        result = subprocess.run(
            [
                LLAMA_BIN,
                "-m",
                MODEL_PATH,
                "-p",
                PROMPT,
                "-n",
                str(MAX_TOKENS),
                "--temp",
                "0.7",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        success = True
        output_text = result.stdout.strip()
    except Exception as e:
        success = False
        output_text = str(e)

    end_time = time.time()
    ram_after = process.memory_info().rss

    return {
        "success": success,
        "latency_sec": round(end_time - start_time, 3),
        "ram_used_mb": round((ram_after - ram_before) / (1024**2), 2),
        "output_length_chars": len(output_text),
        "sample_output": output_text[:500],
    }


############################################
# MAIN
############################################


def main():
    data = {}

    data["model"] = {
        "name": MODEL_NAME,
        "format": MODEL_FORMAT,
        "quantization": QUANTIZATION,
        "disk_size_mb": get_model_size_mb(MODEL_PATH),
    }

    data["system"] = get_system_info()

    if MODEL_FORMAT == "GGUF":
        inference = run_llama_inference()
        data["inference"] = inference
    else:
        data["inference"] = {
            "success": False,
            "reason": "HF inference not executed (resource constrained)",
        }

    # Explicit limitation logging
    data["limitations"] = [
        "FP16 merged model unavailable due to lack of GPU memory",
        "Quantization comparison limited to available formats",
        "Microcontroller deployment infeasible due to memory constraints",
    ]

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[OK] Metrics saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
