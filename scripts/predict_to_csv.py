import re
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["no", "intrinsic", "extrinsic"]
PAT = re.compile(r"\b(no|intrinsic|extrinsic)\b", re.IGNORECASE)


def build_prompt(row: pd.Series) -> str:
    return (
        "Bạn là bộ phân loại ảo giác. Nhiệm vụ: dựa trên NGỮ CẢNH và CÂU TRẢ LỜI của trợ lý, "
        "hãy phân loại duy nhất một nhãn trong {no, intrinsic, extrinsic}.\n\n"
        "Định nghĩa:\n"
        "- no: câu trả lời được hỗ trợ đầy đủ bởi ngữ cảnh.\n"
        "- intrinsic: mâu thuẫn/sai lệch so với ngữ cảnh (sai số liệu, thuộc tính, quan hệ...).\n"
        "- extrinsic: đưa thêm thông tin không có trong ngữ cảnh và không thể kiểm chứng từ ngữ cảnh.\n\n"
        f"NGỮ CẢNH:\n{row['context']}\n\n"
        f"CÂU HỎI:\n{row['prompt']}\n\n"
        f"CÂU TRẢ LỜI CỦA TRỢ LÝ:\n{row['response']}\n\n"
        "Chỉ in đúng một nhãn: no hoặc intrinsic hoặc extrinsic."
    )


@torch.inference_mode()
def predict_csv(input_csv: str, output_csv: str, model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)

    df = pd.read_csv(input_csv)

    ids = []
    preds = []

    pbar = tqdm(range(len(df)), desc="Predict", dynamic_ncols=True)
    for idx in pbar:
        row = df.iloc[idx]
        ids.append(row["id"])  # bắt buộc có cột id

        user_prompt = build_prompt(row)
        messages = [{"role": "user", "content": user_prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
        )

        dec = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = PAT.search(dec)
        y = m.group(1).lower() if m else "no"
        if y not in LABELS:
            y = "no"
        preds.append(y)

    out_df = pd.DataFrame({"id": ids, "predict_label": preds})
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Wrote predictions to {output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory to use for generation")
    parser.add_argument("--input_csv", required=True, help="Input CSV with columns: id, context, prompt, response")
    parser.add_argument("--output_csv", required=True, help="Path to write predictions CSV")
    args = parser.parse_args()

    predict_csv(args.input_csv, args.output_csv, args.ckpt)


