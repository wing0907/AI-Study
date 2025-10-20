# -*- coding: utf-8 -*-
"""
GPT-1 style pipeline (All-in-One):
  (1) BPE tokenizer (~40k) training
  (2) Decoder-only Transformer LM (GPT-2 config) pretraining
  (3) Task-aware finetuning:
      - GLUE classification/regression (sst2/cola/mrpc/stsb/qqp/qnli/rte/mnli)
      - Multiple-Choice QA (RACE, StoryCloze) with GPT2DoubleHeadsModel
Notes:
  - stage 미지정 시, 기본으로 'pretrain' 실행 (VS Code ▶️ 버튼 대비)
  - Windows/VS Code에서 그대로 사용 가능
"""

import os, sys, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------
# Hugging Face & Tokenizers
# --------------------------------
from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers, normalizers
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification, GPT2DoubleHeadsModel,
    PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorWithPadding,
    DataCollatorForLanguageModeling, set_seed
)
from datasets import load_dataset


SPECIALS = {"pad": "<pad>", "bos": "<s>", "eos": "</s>", "sep": "$", "unk": "<unk>"}


# --------------------------------
# Utils
# --------------------------------
def seed_everything(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def info(msg: str):
    print(f"[INFO] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")


# --------------------------------
# 1) Train BPE tokenizer (~40k)
# --------------------------------
def train_bpe_tokenizer(corpus_dir: str, out_dir: str, vocab_size: int = 40000):
    ensure_dir(out_dir)
    files = []
    for root, _, fnames in os.walk(corpus_dir):
        for f in fnames:
            if f.lower().endswith(".txt"):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No .txt files under: {corpus_dir}")

    tok = Tokenizer(models.BPE(unk_token=SPECIALS["unk"]))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[SPECIALS["pad"], SPECIALS["bos"], SPECIALS["eos"], SPECIALS["sep"], SPECIALS["unk"]],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train(files=files, trainer=trainer)

    ensure_dir(out_dir)
    tok_path = os.path.join(out_dir, "gpt1-bpe.json")
    tok.save(tok_path)

    fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        bos_token=SPECIALS["bos"], eos_token=SPECIALS["eos"],
        sep_token=SPECIALS["sep"], pad_token=SPECIALS["pad"], unk_token=SPECIALS["unk"],
    )
    fast.save_pretrained(out_dir)
    info(f"Saved tokenizer to: {out_dir}")


# --------------------------------
# 2) Pretraining dataset (causal LM)
# --------------------------------
class CausalTextBlocks(Dataset):
    """
    Join all docs with </s>, then chunk into fixed token windows (block_size).
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast, text_files: List[str], block_size: int = 512):
        texts = []
        for fp in text_files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read().strip())
        joined = f" {tokenizer.eos_token} ".join(texts)
        enc = tokenizer(joined, return_attention_mask=False, return_tensors=None)
        ids = enc["input_ids"]
        self.examples = []
        for i in range(0, len(ids) - block_size, block_size):
            self.examples.append(ids[i:i+block_size])
        self.tokenizer = tokenizer

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        x = self.examples[i]
        return {"input_ids": torch.tensor(x, dtype=torch.long)}

def build_pretrain_dataset(tokenizer_dir: str, corpus_dir: str, block_size: int = 512) -> CausalTextBlocks:
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    files = []
    for root, _, fnames in os.walk(corpus_dir):
        for f in fnames:
            if f.lower().endswith(".txt"):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No .txt files under: {corpus_dir}")
    return CausalTextBlocks(tok, files, block_size)

def build_gpt1_config(tokenizer_dir: str,
                      n_layer=12, n_head=12, n_embd=768, n_positions=512, ffn_mult=4):
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    cfg = GPT2Config(
        vocab_size=len(tok),
        n_positions=n_positions, n_ctx=n_positions,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        n_inner=n_embd * ffn_mult,  # 768*4 = 3072
        activation_function="gelu_new",
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id
    )
    return cfg

def pretrain(args):
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    assert tok.pad_token is not None, "pad token required (check tokenizer training)"

    cfg = build_gpt1_config(args.tokenizer_dir, args.n_layer, args.n_head, args.n_embd, args.n_positions)
    model = GPT2LMHeadModel(cfg)

    ds = build_pretrain_dataset(args.tokenizer_dir, args.corpus_dir, args.block_size)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        logging_steps=50, save_steps=args.save_steps,
        bf16=args.bf16, fp16=not args.bf16,
        report_to="none"
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    info("Pretraining done. Model + tokenizer saved.")


# --------------------------------
# 3) Task-aware finetuning (GLUE & MCQA)
# --------------------------------
@dataclass
class TaskExample:
    text: str
    label: Optional[float]

class TaskDataset(Dataset):
    def __init__(self, tokenizer, examples: List[TaskExample], max_length: int = 512):
        self.tok = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        enc = self.tok(ex.text, max_length=self.max_length, truncation=True, padding=False, add_special_tokens=True)
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        if ex.label is not None:
            if isinstance(ex.label, float):
                item["labels"] = torch.tensor(ex.label, dtype=torch.float)
            else:
                item["labels"] = torch.tensor(int(ex.label), dtype=torch.long)
        return item

def format_entailment(premise: str, hypothesis: str, tok) -> str:
    return f"{tok.bos_token}{premise} {tok.sep_token} {hypothesis}{tok.eos_token}"

def format_similarity(s1: str, s2: str, tok, use_both_orders=False) -> List[str]:
    a = f"{tok.bos_token}{s1} {tok.sep_token} {s2}{tok.eos_token}"
    if use_both_orders:
        b = f"{tok.bos_token}{s2} {tok.sep_token} {s1}{tok.eos_token}"
        return [a, b]
    return [a]

def format_mcqa(context: str, question: str, choices: List[str], tok) -> List[str]:
    base = f"{tok.bos_token}{context} {tok.sep_token} {question} {tok.sep_token} "
    return [base + c + tok.eos_token for c in choices]

def load_task_dataset(task: str, split: str, tokenizer, use_both_orders=False) -> Tuple[Dataset, Dict]:
    """
    Returns (dataset, meta). meta may include: num_labels, is_regression, multiple_choice
    """
    task = task.lower()
    meta = {"num_labels": None, "is_regression": False, "multiple_choice": False}

    # GLUE
    if task in {"sst2", "cola", "mrpc", "stsb", "qqp", "qnli", "rte", "mnli"}:
        ds = load_dataset("glue", task)
        d = ds[split if task != "mnli" else ("validation_matched" if split == "validation" else "train")]
        exs = []

        if task in {"sst2", "cola"}:
            key = "sentence"
            for r in d:
                txt = f"{tokenizer.bos_token}{r[key]}{tokenizer.eos_token}"
                exs.append(TaskExample(txt, int(r["label"])))
            meta.update(num_labels=2)

        elif task in {"mrpc", "qqp", "stsb"}:
            a1, a2 = "sentence1", "sentence2"
            for r in d:
                seqs = format_similarity(r[a1], r[a2], tokenizer, use_both_orders=use_both_orders)
                label = r["label"]
                if task == "stsb":
                    meta["is_regression"] = True
                    label = float(label)
                for s in seqs:
                    exs.append(TaskExample(s, label))
            meta.update(num_labels=1 if task == "stsb" else 2)

        elif task in {"qnli", "rte", "mnli"}:
            for r in d:
                if task == "qnli":
                    prem, hyp = r["question"], r["sentence"]
                elif task == "rte":
                    prem, hyp = r["sentence1"], r["sentence2"]
                else:
                    prem, hyp = r["premise"], r["hypothesis"]
                exs.append(TaskExample(format_entailment(prem, hyp, tokenizer), int(r["label"])))
            meta.update(num_labels=3 if task == "mnli" else 2)

        return TaskDataset(tokenizer, exs), meta

    # RACE
    if task == "race":
        ds = load_dataset("race", "all")[split]
        return ds, {"multiple_choice": True}

    # StoryCloze (2016)
    if task in {"story_cloze", "storycloze"}:
        ds = load_dataset("story_cloze", "2016")[split]  # 'validation'에는 라벨 있음
        return ds, {"multiple_choice": True}

    raise ValueError(f"Unknown task: {task}")


# ---- GLUE finetune
def finetune_glue(args):
    seed_everything(args.seed)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": SPECIALS["pad"]})

    train_ds, _ = load_task_dataset(args.task, "train", tok, use_both_orders=True)
    eval_ds,  _ = load_task_dataset(args.task, "validation", tok, use_both_orders=False)

    # base model
    model = GPT2ForSequenceClassification.from_pretrained(args.pretrained_or_dir, num_labels=2)
    model.resize_token_embeddings(len(tok))

    is_reg = (args.task.lower() == "stsb")
    if is_reg:
        model.config.num_labels = 1
        model.config.problem_type = "regression"

    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8 if (args.bf16 or torch.cuda.is_available()) else None)

    from evaluate import load as load_metric
    acc_metric = load_metric("accuracy")
    f1_metric  = load_metric("f1")
    pearson    = load_metric("pearsonr")

    def compute_metrics(pred):
        if is_reg:
            preds = pred.predictions.reshape(-1)
            return {"pearson": pearson.compute(predictions=preds, references=pred.label_ids)["pearson"]}
        preds = np.argmax(pred.predictions, axis=-1)
        out = {"accuracy": acc_metric.compute(predictions=preds, references=pred.label_ids)["accuracy"]}
        try:
            out["f1"] = f1_metric.compute(predictions=preds, references=pred.label_ids)["f1"]
        except Exception:
            pass
        return out

    targs = TrainingArguments(
        output_dir=args.output_dir, overwrite_output_dir=True,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum, warmup_ratio=0.002,
        lr_scheduler_type="linear",
        logging_steps=50, evaluation_strategy="steps", eval_steps=200, save_steps=200,
        load_best_model_at_end=True, metric_for_best_model="pearson" if is_reg else "accuracy",
        bf16=args.bf16, fp16=not args.bf16, report_to="none"
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    info("Finetuning(GLUE) done.")


# ---- MCQA finetune (RACE / StoryCloze)
def finetune_mcqa(args):
    seed_everything(args.seed)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": SPECIALS["pad"]})

    model = GPT2DoubleHeadsModel.from_pretrained(args.pretrained_or_dir)
    model.resize_token_embeddings(len(tok))

    ds_train, _ = load_task_dataset(args.task, "train", tok)
    ds_eval,  _ = load_task_dataset(args.task, "validation", tok)

    def _race_item(batch):
        ctx, q, opts, ans = batch["article"], batch["question"], batch["options"], batch["answer"]
        if isinstance(ans, str) and len(ans) == 1 and ans.isalpha():
            label_idx = ord(ans.upper()) - ord('A')
        else:
            try: label_idx = opts.index(ans)
            except Exception: label_idx = -1
        seqs = format_mcqa(ctx, q, opts, tok)
        return seqs, label_idx

    def _story_item(batch):
        ctx_parts = []
        for key in ["input_sentence_1", "input_sentence_2", "input_sentence_3"]:
            if key in batch and isinstance(batch[key], str): ctx_parts.append(batch[key])
        if not ctx_parts:
            for key in ["sentence1", "sentence2", "sentence3"]:
                if key in batch and isinstance(batch[key], str): ctx_parts.append(batch[key])
        ctx = " ".join(ctx_parts)
        q = ""
        opts = []
        for key in ["sentence_quiz1", "sentence_quiz2"]:
            if key in batch and isinstance(batch[key], str): opts.append(batch[key])
        label_idx = int(batch["answer_right_ending"]) - 1 if "answer_right_ending" in batch else -1
        seqs = format_mcqa(ctx, q, opts, tok)
        return seqs, label_idx

    def preprocess_mc(batch):
        return _race_item(batch) if args.task == "race" else _story_item(batch)

    class MCQADataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds): self.hf = hf_ds
        def __len__(self): return len(self.hf)
        def __getitem__(self, idx):
            seqs, label_idx = preprocess_mc(self.hf[int(idx)])
            enc = tok(seqs, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
            return {
                "input_ids": enc["input_ids"],                 # [C, L]
                "attention_mask": enc["attention_mask"],       # [C, L]
                "labels": torch.tensor(label_idx, dtype=torch.long) if label_idx is not None and label_idx >= 0 else None
            }

    train_ds = MCQADataset(ds_train)
    eval_ds  = MCQADataset(ds_eval)

    def collate_mc(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)      # [B, C, L]
        attn      = torch.stack([b["attention_mask"] for b in batch], dim=0) # [B, C, L]

        pad_id = tok.pad_token_id
        eos_id = tok.eos_token_id
        B, C, L = input_ids.size()
        mc_token_ids = torch.zeros((B, C), dtype=torch.long)
        for i in range(B):
            for j in range(C):
                seq = input_ids[i, j]
                eos_pos = (seq == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    idx = int(eos_pos[-1])
                else:
                    nonpad = (seq != pad_id).nonzero(as_tuple=False)
                    idx = int(nonpad[-1]) if nonpad.numel() > 0 else L - 1
                mc_token_ids[i, j] = idx

        labels = None
        if batch[0]["labels"] is not None:
            labels = torch.stack([b["labels"] for b in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "mc_token_ids": mc_token_ids,
            "mc_labels": labels
        }

    class MCTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("mc_labels", None)
            outputs = model(**inputs, mc_labels=labels)
            loss = outputs.mc_loss
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            labels = inputs.get("mc_labels", None)
            with torch.no_grad():
                outputs = model(**{k:v for k,v in inputs.items() if k != "mc_labels"}, mc_labels=labels)
            loss = outputs.mc_loss if labels is not None else None
            return (loss, outputs.mc_logits, labels)

    from evaluate import load as load_metric
    acc_metric = load_metric("accuracy")
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": acc_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]}

    targs = TrainingArguments(
        output_dir=args.output_dir, overwrite_output_dir=True,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=max(1, args.batch_size // 2),
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum, warmup_ratio=0.002,
        logging_steps=50, evaluation_strategy="steps", eval_steps=200, save_steps=200,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        lr_scheduler_type="linear",
        bf16=args.bf16, fp16=not args.bf16, report_to="none"
    )

    trainer = MCTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tok, data_collator=collate_mc, compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    info("Finetuning(MCQA) done.")


# --------------------------------
# CLI (stage default = 'pretrain')
# --------------------------------
def build_argparser():
    p = argparse.ArgumentParser("GPT-1 style pretrain + finetune")
    sub = p.add_subparsers(dest="stage")  # required=False -> 기본 stage 수동 처리

    # tokenizer
    t = sub.add_parser("train_tokenizer")
    t.add_argument("--corpus_dir", type=str, required=True)
    t.add_argument("--out_dir", type=str, required=True)
    t.add_argument("--vocab_size", type=int, default=40000)

    # pretrain
    pr = sub.add_parser("pretrain")
    pr.add_argument("--tokenizer_dir", type=str, required=True)
    pr.add_argument("--corpus_dir", type=str, required=True)
    pr.add_argument("--output_dir", type=str, required=True)
    pr.add_argument("--block_size", type=int, default=512)
    pr.add_argument("--epochs", type=int, default=100)
    pr.add_argument("--lr", type=float, default=2.5e-4)
    pr.add_argument("--warmup_steps", type=int, default=2000)
    pr.add_argument("--batch_size", type=int, default=4)
    pr.add_argument("--eval_batch_size", type=int, default=8)
    pr.add_argument("--grad_accum", type=int, default=4)
    pr.add_argument("--weight_decay", type=float, default=0.01)
    pr.add_argument("--save_steps", type=int, default=1000)
    pr.add_argument("--bf16", action="store_true")
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--n_layer", type=int, default=12)
    pr.add_argument("--n_head", type=int, default=12)
    pr.add_argument("--n_embd", type=int, default=768)
    pr.add_argument("--n_positions", type=int, default=512)

    # GLUE
    ft = sub.add_parser("finetune_glue")
    ft.add_argument("--tokenizer_dir", type=str, required=True)
    ft.add_argument("--pretrained_or_dir", type=str, required=True)
    ft.add_argument("--task", type=str, required=True, help="sst2|cola|mrpc|stsb|qqp|qnli|rte|mnli")
    ft.add_argument("--output_dir", type=str, required=True)
    ft.add_argument("--epochs", type=int, default=3)
    ft.add_argument("--lr", type=float, default=6.25e-5)
    ft.add_argument("--batch_size", type=int, default=8)
    ft.add_argument("--eval_batch_size", type=int, default=16)
    ft.add_argument("--grad_accum", type=int, default=1)
    ft.add_argument("--bf16", action="store_true")
    ft.add_argument("--seed", type=int, default=42)

    # MCQA
    mc = sub.add_parser("finetune_mcqa")
    mc.add_argument("--tokenizer_dir", type=str, required=True)
    mc.add_argument("--pretrained_or_dir", type=str, required=True)
    mc.add_argument("--task", type=str, required=True, help="race|story_cloze")
    mc.add_argument("--output_dir", type=str, required=True)
    mc.add_argument("--epochs", type=int, default=3)
    mc.add_argument("--lr", type=float, default=6.25e-5)
    mc.add_argument("--batch_size", type=int, default=4)
    mc.add_argument("--eval_batch_size", type=int, default=8)
    mc.add_argument("--grad_accum", type=int, default=1)
    mc.add_argument("--bf16", action="store_true")
    mc.add_argument("--seed", type=int, default=42)
    mc.add_argument("--max_length", type=int, default=512)

    return p

def parse_args_with_default_stage() -> argparse.Namespace:
    ap = build_argparser()
    args, unknown = ap.parse_known_args()
    if args.stage is None:
        warn("No 'stage' specified. Defaulting to 'pretrain'. (Use train_tokenizer|pretrain|finetune_glue|finetune_mcqa)")
        # 최소 인자 체크
        # pretrain에 필요한 필수 인자가 없으면 친절히 메시지
        # 여기서는 --tokenizer_dir/--corpus_dir/--output_dir를 요구
        # 자동 기본값은 넣지 않습니다.
        # (원하면 아래를 사용자 환경에 맞게 하드코딩해도 됨)
        # 예: args.tokenizer_dir = "d:/..."; 등
        args.stage = "pretrain"
        if not getattr(args, "tokenizer_dir", None) or not getattr(args, "corpus_dir", None) or not getattr(args, "output_dir", None):
            ap.parse_args(["pretrain", "-h"])  # 도움말 출력 후 종료
    return ap.parse_args()

def main():
    args = parse_args_with_default_stage()

    if args.stage == "train_tokenizer":
        train_bpe_tokenizer(args.corpus_dir, args.out_dir, args.vocab_size)

    elif args.stage == "pretrain":
        pretrain(args)

    elif args.stage == "finetune_glue":
        finetune_glue(args)

    elif args.stage == "finetune_mcqa":
        finetune_mcqa(args)

    else:
        print("Unknown stage:", args.stage)
        sys.exit(2)

if __name__ == "__main__":
    main()
