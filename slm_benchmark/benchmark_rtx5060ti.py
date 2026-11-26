import gc
import time
import torch
import pynvml
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def read_energy_mj(handle):
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle), True
    except pynvml.NVMLError:
        return pynvml.nvmlDeviceGetPowerUsage(handle), False

MODELS = [
    {"tier": "FP16 (Dense)",        "repo": "microsoft/Phi-3.5-mini-instruct", "mode": "fp16"},
    {"tier": "INT8 (BitsAndBytes)", "repo": "microsoft/Phi-3.5-mini-instruct", "mode": "int8"},
    {"tier": "INT4 (NF4 4-Bit)",    "repo": "microsoft/Phi-3.5-mini-instruct", "mode": "int4"},
]

PROMPT = (
    "Explain how memory bandwidth limits autoregressive Transformer decoding "
    "and why weight quantization reduces per-token latency on edge GPUs."
)
GEN_TOKENS = 256
NUM_TRIALS = 10

def load_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo"])
    if cfg["mode"] == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["repo"], dtype=torch.float16, device_map="cuda"
        )
    elif cfg["mode"] == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["repo"], quantization_config=bnb_config, device_map="cuda"
        )
    elif cfg["mode"] == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["repo"], quantization_config=bnb_config, device_map="cuda"
        )
    model.eval()
    return model, tokenizer

results = []

for cfg in MODELS:
    print(f"\n=== Benchmarking {cfg['tier']} ===")
    torch.cuda.empty_cache()
    gc.collect()

    model, tokenizer = load_model_and_tokenizer(cfg)
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()

    for trial in range(1, NUM_TRIALS + 1):
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        e_start, has_counter = read_energy_mj(gpu_handle)
        t_start = time.perf_counter()

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                min_new_tokens=GEN_TOKENS,
                max_new_tokens=GEN_TOKENS,
                do_sample=False,
            )

        torch.cuda.synchronize()
        total_time_s = time.perf_counter() - t_start
        e_end, _ = read_energy_mj(gpu_handle)

        decode_ms = (total_time_s * 1000.0) - ttft_ms
        latency_ms_tok = decode_ms / (GEN_TOKENS - 1)
        throughput_tok_s = 1000.0 / latency_ms_tok

        if has_counter:
            total_joules = (e_end - e_start) / 1000.0
            avg_power_w = total_joules / total_time_s
        else:
            avg_power_w = ((e_start + e_end) / 2.0) / 1000.0
            total_joules = avg_power_w * total_time_s

        energy_j_tok = total_joules / GEN_TOKENS
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        results.append({
            "Precision": cfg["tier"],
            "Trial": trial,
            "Peak_VRAM_GB": round(peak_vram_gb, 3),
            "TTFT_ms": round(ttft_ms, 2),
            "Latency_ms_per_tok": round(latency_ms_tok, 2),
            "Throughput_tok_per_s": round(throughput_tok_s, 1),
            "Avg_Power_W": round(avg_power_w, 1),
            "Energy_J_per_tok": round(energy_j_tok, 4),
        })

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

pynvml.nvmlShutdown()

df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print("\n=== SUMMARY FOR TABLE I ===")
print(df.groupby("Precision").mean(numeric_only=True).to_string())
