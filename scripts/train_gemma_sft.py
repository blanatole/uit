import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


SYSTEM_PROMPT = "Bạn là bộ phân loại ảo giác. Nhiệm vụ: dựa trên NGỮ CẢNH và CÂU TRẢ LỜI của trợ lý, hãy phân loại duy nhất một nhãn trong {no, intrinsic, extrinsic}.\n\nĐịnh nghĩa:\n- no: câu trả lời được hỗ trợ đầy đủ bởi ngữ cảnh.\n- intrinsic: mâu thuẫn/sai lệch so với ngữ cảnh (sai số liệu, thuộc tính, quan hệ...).\n- extrinsic: đưa thêm thông tin không có trong ngữ cảnh và không thể kiểm chứng từ ngữ cảnh.\n\nChỉ in đúng một nhãn: no hoặc intrinsic hoặc extrinsic."


def build_chat(tokenizer, context: str, prompt: str, response: str, label: str) -> str:
    user = (
        "NGỮ CẢNH:\n" + context + "\n\n" +
        "CÂU HỎI:\n" + prompt + "\n\n" +
        "CÂU TRẢ LỜI CỦA TRỢ LÝ:\n" + response + "\n\n" +
        "Chỉ in đúng một nhãn: no hoặc intrinsic hoặc extrinsic."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": label},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def get_dataset(tokenizer, train_path: Path, val_path: Path, max_length: int):
    def _load(split_path: Path):
        data = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                context = ex.get('context', '')
                prompt = ex.get('prompt', '')
                response = ex.get('response', '')
                label = ex.get('label', '')
                text = build_chat(tokenizer, context, prompt, response, label)
                data.append({"text": text})
        return data

    train_data = _load(train_path)
    val_data = _load(val_path)
    from datasets import Dataset
    ds_train = Dataset.from_list(train_data)
    ds_val = Dataset.from_list(val_data)

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    return ds_train.map(tokenize_fn, batched=True, remove_columns=["text"]), ds_val.map(tokenize_fn, batched=True, remove_columns=["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-7b-it")
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "outputs" / "gemma-sft"))
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    assert train_path.exists() and val_path.exists(), "Missing train/val jsonl. Run preprocessing first."

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_train, ds_val = get_dataset(tokenizer, train_path, val_path, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(args.model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to=["none"],
        fp16=args.fp16,
        bf16=args.bf16,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()


