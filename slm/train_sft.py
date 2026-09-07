"""QLoRA SFT of a small instruct model on the SAS-reconcile debugging task.

Run (from repo root, inside the .venv):

    .venv/bin/python -m slm.train_sft \
        --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --data_path data/generated/sft.jsonl \
        --out_dir checkpoints/sft-0.5b

Designed for a 16 GB CUDA GPU (Blackwell,smi / capability 12.0) with bf16 +
4-bit NF4 QLoRA.  We do NOT use vLLM (Blackwell compatibility risk).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset

from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class CausalLMCollator:
    """Pad input_ids / attention_mask / labels together for CAUSAL_LM training.

    ``labels`` are padded with -100 so padding tokens are ignored by the loss.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        max_len = max(len(f["input_ids"]) for f in features)
        pad = self.pad_token_id
        batch = {k: [] for k in ("input_ids", "attention_mask", "labels")}
        for f in features:
            n = len(f["input_ids"])
            m = max_len - n
            batch["input_ids"].append(f["input_ids"] + [pad] * m)
            batch["attention_mask"].append(f["attention_mask"] + [0] * m)
            batch["labels"].append(list(f["labels"]) + [-100] * m)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
    TrainingArguments,
)

DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = "data/generated/sft.jsonl"


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True,
                                        use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_4bit_model(name: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def make_lora(r: int, alpha: int, dropout: float):
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def build_dataset(path: Path, tokenizer, max_len: int) -> Dataset:
    """Tokenize the ``train`` split with prompt-suffix loss masking."""
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    records = [r for r in records if r["split"] == "train"]

    def enc(rec):
        sys_, user, asst = rec["system"], rec["user"], rec["assistant"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        eos = tokenizer.eos_token
        full = prompt + asst + (eos or "\n")
        full_ids = tokenizer(full, truncation=True, max_length=max_len)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
        labels = list(full_ids["input_ids"])
        n_prompt = len(prompt_ids)
        # mask the prompt (and any padding beyond max_len)
        for i in range(min(n_prompt, len(labels))):
            labels[i] = -100
        return {
            "input_ids": full_ids["input_ids"],
            "attention_mask": full_ids["attention_mask"],
            "labels": labels,
        }

    ds = Dataset.from_list([enc(r) for r in records])
    return ds


def train(args) -> None:
    args.out_dir = Path(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.base_model)
    model = load_4bit_model(args.base_model)
    model = get_peft_model(model, make_lora(args.r, args.alpha, args.dropout))
    model.print_trainable_parameters()

    ds = build_dataset(Path(args.data_path), tokenizer, args.max_len)

    tr_args = TrainingArguments(
        output_dir=str(args.out_dir / "run"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="tensorboard" if args.tensorboard else "none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    collator = CausalLMCollator(tokenizer.pad_token_id)
    trainer = Trainer(
        model=model, args=tr_args, train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    # Save the LoRA adapter + a merged fp16/bfloat16 copy for inference.
    model.save_pretrained(str(args.out_dir / "lora"))
    tokenizer.save_pretrained(str(args.out_dir / "lora"))
    adapter_weights = Path(args.out_dir) / "lora" / "adapter_model.safetensors"
    if adapter_weights.exists():
        print(f"saved LoRA adapter: {adapter_weights} ({adapter_weights.stat().st_size//1024} KiB)")

    final_metrics = trainer.evaluate() if False else trainer.state.log_history[-1] if trainer.state.log_history else {}
    print("training complete; final log:", json.dumps(final_metrics)[:500])


def main():
    ap = argparse.ArgumentParser(description="QLoRA SFT for the SAS-reconcile SLM.")
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--data_path", default=DEFAULT_DATA)
    ap.add_argument("--out_dir", default="checkpoints/sft-0.5b")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tensorboard", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
