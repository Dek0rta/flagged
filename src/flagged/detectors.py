"""Open-source AI-text detectors, all scored so that higher = "more likely AI".

Supervised classifiers (year released):
  openai_2019     openai-community/roberta-base-openai-detector (GPT-2 output detector)
  hc3_2023        Hello-SimpleAI/chatgpt-detector-roberta (trained on HC3)
  radar_2023      TrustSafeAI/RADAR-Vicuna-7B (adversarially trained, NeurIPS 2023)
  superannotate_2024  SuperAnnotate/ai-detector (RoBERTa-large, RAID + custom data)
  fakespot_2025   fakespot-ai/roberta-base-ai-text-detection-v1 (Mozilla, used in a Firefox add-on)
  desklib_2025    desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, top of RAID leaderboard)

Zero-shot detectors (no training data, read the text with a language model):
  logppl          log-perplexity under Qwen2.5-1.5B (the signal Liang et al. blamed)
  binoculars      Hans et al. 2024, observer Qwen2.5-1.5B / performer Qwen2.5-1.5B-Instruct
  fastdetect_qwen Fast-DetectGPT (Bao et al. 2024), analytic, Qwen2.5-1.5B as sampler and scorer
  fastdetect_neo  Fast-DetectGPT white-box setting from the paper, GPT-Neo-2.7B
"""

import re
from html import unescape

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer, RobertaModel)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MAX_TOKENS = 512


def batches(texts, size):
    """Yield (indices, texts) batches, longest first, to keep padding low."""
    order = np.argsort([-len(t) for t in texts])
    for start in range(0, len(order), size):
        idx = order[start:start + size]
        yield idx, [texts[i] for i in idx]


def fakespot_clean(t: str) -> str:
    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)
    t = re.sub(r"`[^`]*`", "", t)
    t = re.sub(r"(\*\*|__)(.*?)\1", r"\2", t)
    t = re.sub(r"#+ ", "", t)
    t = re.sub(r"<.*?>", "", t)
    t = unescape(t)
    for ch in ("\n", "\t", "\r"):
        t = t.replace(ch, " ")
    t = t.replace(" ,", ",")
    return re.sub(" +", " ", t)


# ---------------------------------------------------------------- supervised

class Classifier:
    """A Hugging Face sequence classifier; returns P(AI)."""

    def __init__(self, model_id, ai_index, preprocess=None, batch_size=16):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval()
        self.ai_index, self.preprocess, self.batch_size = ai_index, preprocess, batch_size

    def logits(self, enc):
        return self.model(**enc).logits

    def prob(self, logits):
        return logits.float().softmax(-1)[:, self.ai_index]

    @torch.no_grad()
    def score(self, texts):
        out = np.zeros(len(texts))
        for idx, chunk in tqdm(list(batches(texts, self.batch_size)), leave=False):
            if self.preprocess:
                chunk = [self.preprocess(t) for t in chunk]
            enc = self.tok(chunk, truncation=True, max_length=MAX_TOKENS, padding=True,
                           return_tensors="pt").to(DEVICE)
            out[idx] = self.prob(self.logits(enc)).cpu().numpy()
        return out


class DesklibHead(nn.Module):
    """DeBERTa-v3-large, mean-pooled, one logit (architecture from the model card)."""

    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **_):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).float()
        return self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9))


class SuperAnnotateHead(nn.Module):
    """RoBERTa-large, <s> token, one logit (architecture from the model's repo)."""

    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **_):
        return self.dense(self.roberta(input_ids, attention_mask=attention_mask)[0][:, 0])


class SigmoidClassifier(Classifier):
    def __init__(self, model_id, head, base_config, preprocess=None, batch_size=8):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = head(AutoConfig.from_pretrained(base_config))
        state = load_file(hf_hub_download(model_id, "model.safetensors"))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        assert not unexpected and all("position_ids" in k for k in missing), (missing, unexpected)
        self.model = self.model.to(DEVICE).eval()
        self.preprocess, self.batch_size = preprocess, batch_size

    def logits(self, enc):
        return self.model(enc["input_ids"], enc["attention_mask"])

    def prob(self, logits):
        return logits.float().sigmoid()[:, 0]


def supervised(name):
    if name == "openai_2019":
        return Classifier("openai-community/roberta-base-openai-detector", ai_index=0)   # 0 = "Fake"
    if name == "hc3_2023":
        return Classifier("Hello-SimpleAI/chatgpt-detector-roberta", ai_index=1)        # 1 = "ChatGPT"
    if name == "radar_2023":
        return Classifier("TrustSafeAI/RADAR-Vicuna-7B", ai_index=0)                    # 0 = AI, per RADAR demo
    if name == "superannotate_2024":
        return SigmoidClassifier("SuperAnnotate/ai-detector", SuperAnnotateHead, "FacebookAI/roberta-large",
                                 preprocess=lambda t: " ".join(t.split()))
    if name == "fakespot_2025":
        return Classifier("fakespot-ai/roberta-base-ai-text-detection-v1", ai_index=1,  # 1 = "AI"
                          preprocess=fakespot_clean)
    if name == "desklib_2025":
        return SigmoidClassifier("desklib/ai-text-detector-v1.01", DesklibHead, "desklib/ai-text-detector-v1.01")
    raise KeyError(name)


SUPERVISED = ["openai_2019", "hc3_2023", "radar_2023", "superannotate_2024", "fakespot_2025", "desklib_2025"]


# ---------------------------------------------------------------- zero-shot

def load_lm(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(DEVICE).eval()
    return tok, model


def token_stats(obs_logits, perf_logits, labels, chunk=128):
    """Per-token statistics, computed 128 positions at a time to keep memory low.

    From the observer: its NLL and the analytic Fast-DetectGPT terms (Bao et al. 2024, eq. 4)
    with the observer as both sampling and scoring model. From the performer (if given):
    its NLL and the observer-vs-performer cross-entropy used by Binoculars (Hans et al. 2024).
    """
    out = {k: [] for k in ("nll_obs", "fd_mean", "fd_var", "nll_perf", "x_ent")}
    for s in range(0, labels.shape[1], chunk):
        y = labels[:, s:s + chunk].unsqueeze(-1)
        lo = obs_logits[:, s:s + chunk].float().log_softmax(-1)
        po = lo.exp()
        mean = (po * lo).sum(-1)
        out["nll_obs"].append(-lo.gather(-1, y).squeeze(-1))
        out["fd_mean"].append(mean)
        out["fd_var"].append((po * lo.square()).sum(-1) - mean.square())
        if perf_logits is not None:
            lp = perf_logits[:, s:s + chunk].float().log_softmax(-1)
            out["nll_perf"].append(-lp.gather(-1, y).squeeze(-1))
            out["x_ent"].append(-(po * lp).sum(-1))
    return {k: torch.cat(v, 1) for k, v in out.items() if v}


def fast_detect(st, mask):
    """Sampling discrepancy: (log p(x) - E[log p]) / sqrt(Var[log p]), summed over tokens."""
    return ((-st["nll_obs"] - st["fd_mean"]) * mask).sum(-1) / (st["fd_var"] * mask).sum(-1).sqrt()


def encode(tok, chunk):
    return tok(chunk, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt").to(DEVICE)


@torch.no_grad()
def zero_shot_qwen(texts, batch_size=2):
    """One pass over Qwen2.5-1.5B (+Instruct) gives logppl, Binoculars and Fast-DetectGPT."""
    tok, observer = load_lm("Qwen/Qwen2.5-1.5B")
    _, performer = load_lm("Qwen/Qwen2.5-1.5B-Instruct")
    out = {k: np.zeros(len(texts)) for k in ("logppl", "binoculars", "fastdetect_qwen")}
    for i, (idx, chunk) in enumerate(tqdm(list(batches(texts, batch_size)), leave=False)):
        enc = encode(tok, chunk)
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        n = mask.sum(-1)
        st = token_stats(observer(**enc).logits[:, :-1], performer(**enc).logits[:, :-1], labels)
        log_ppl = (st["nll_perf"] * mask).sum(-1) / n
        log_xppl = (st["x_ent"] * mask).sum(-1) / n
        out["binoculars"][idx] = (-(log_ppl / log_xppl)).cpu().numpy()                 # low B = AI
        out["logppl"][idx] = (-((st["nll_obs"] * mask).sum(-1) / n)).cpu().numpy()     # low ppl = AI
        out["fastdetect_qwen"][idx] = fast_detect(st, mask).cpu().numpy()
        if i % 50 == 0:
            torch.mps.empty_cache()
    return out


@torch.no_grad()
def zero_shot_neo(texts, batch_size=2):
    tok, model = load_lm("EleutherAI/gpt-neo-2.7B")
    out = np.zeros(len(texts))
    for i, (idx, chunk) in enumerate(tqdm(list(batches(texts, batch_size)), leave=False)):
        enc = encode(tok, chunk)
        st = token_stats(model(**enc).logits[:, :-1], None, enc["input_ids"][:, 1:])
        out[idx] = fast_detect(st, enc["attention_mask"][:, 1:].float()).cpu().numpy()
        if i % 50 == 0:
            torch.mps.empty_cache()
    return {"fastdetect_neo": out}


ZERO_SHOT = ["logppl", "binoculars", "fastdetect_qwen", "fastdetect_neo"]
ALL = SUPERVISED + ZERO_SHOT
