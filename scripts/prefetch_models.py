"""Download every model the pipeline needs (weights only in safetensors where available)."""

from huggingface_hub import snapshot_download

MODELS = [
    ("SuperAnnotate/ai-detector", None),
    ("FacebookAI/roberta-large", ["config.json"]),
    ("fakespot-ai/roberta-base-ai-text-detection-v1", ["*.json", "*.txt", "model.safetensors"]),
    ("desklib/ai-text-detector-v1.01", None),
    ("Qwen/Qwen2.5-1.5B", ["*.json", "*.txt", "*.safetensors"]),
    ("Qwen/Qwen2.5-1.5B-Instruct", ["*.json", "*.txt", "*.safetensors"]),
    ("EleutherAI/gpt-neo-2.7B", ["*.json", "*.txt", "model.safetensors"]),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit", None),
]

for repo, patterns in MODELS:
    path = snapshot_download(repo, allow_patterns=patterns)
    print("ok", repo, path, flush=True)
