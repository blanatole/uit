import re
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["no", "intrinsic", "extrinsic"]
# Siết regex: chỉ bắt ở đầu chuỗi, có thể có "label:"
PAT = re.compile(r"^\s*(?:label\s*:\s*)?(no|intrinsic|extrinsic)\b", re.IGNORECASE)

ROLE_SYS = {
    "literalist": (
        "Vai trò: BÁM VĂN BẢN. Bạn cẩn thận, chính xác, chỉ dựa vào thông tin có trong NGỮ CẢNH.\n"
        "Nếu câu trả lời KHÔNG có trong ngữ cảnh → extrinsic\n"
        "Nếu câu trả lời SAI so với ngữ cảnh → intrinsic\n"
        "Nếu câu trả lời ĐÚNG và có trong ngữ cảnh → no"
    ),
    "skeptic": (
        "Vai trò: HOÀI NGHI. Bạn tìm kiếm lỗi, mâu thuẫn, thiếu chứng cứ.\n"
        "Nếu thấy thông tin KHÔNG có trong ngữ cảnh → extrinsic\n"
        "Nếu thấy thông tin SAI so với ngữ cảnh → intrinsic\n"
        "Nếu thấy thông tin ĐÚNG và có chứng cứ → no"
    ),
    "verifier": (
        "Vai trò: THẨM ĐỊNH. Bạn kiểm tra từng mệnh đề một cách khách quan.\n"
        "Đối chiếu từng câu với NGỮ CẢNH:\n"
        "- Có trong ngữ cảnh và đúng → no\n"
        "- Có trong ngữ cảnh nhưng sai → intrinsic\n"
        "- Không có trong ngữ cảnh → extrinsic"
    ),
}

JUDGE_SYS = (
    "Bạn là TRỌNG TÀI cuối cùng. Nhận votes từ 3 vai trò chuyên gia.\n\n"
    "QUY TẮC QUYẾT ĐỊNH:\n"
    "1. Nếu 2+ vai trò vote 'no' → no (câu trả lời đúng)\n"
    "2. Nếu 2+ vai trò vote 'intrinsic' → intrinsic (sai so với ngữ cảnh)\n"
    "3. Nếu 2+ vai trò vote 'extrinsic' → extrinsic (thêm thông tin không có)\n"
    "4. Nếu hòa (1-1-1) → chọn 'no' (ưu tiên an toàn)\n\n"
    "Chỉ in 1 từ duy nhất: no hoặc intrinsic hoặc extrinsic."
)


def build_case(row: pd.Series) -> str:
    return (
        f"NGỮ CẢNH:\n{row['context']}\n\n"
        f"CÂU HỎI:\n{row['prompt']}\n\n"
        f"CÂU TRẢ LỜI CỦA TRỢ LÝ:\n{row['response']}\n"
    )


def parse_label(text: str):
    m = PAT.search(text)
    return m.group(1).lower() if m else None


@torch.inference_mode()
def debate_predict_csv(input_csv: str, output_csv: str, model_dir: str, rounds: int = 1):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)

    # Đọc JSONL thay vì CSV
    df = pd.read_json(input_csv, lines=True)

    ids = []
    preds = []

    # Test mode: chỉ dùng 20 mẫu đầu để kiểm tra
    df_test = df.head(20)
    print(f"🧪 Test mode: Using only {len(df_test)} samples")
    
    pbar = tqdm(range(len(df_test)), desc="DebateCSV", dynamic_ncols=True)
    for i in pbar:
        row = df_test.iloc[i]
        ids.append(row["id"])
        case = build_case(row)

        votes = []
        for _ in range(rounds):
            for role in ["literalist", "skeptic", "verifier"]:
                msgs = [
                    {"role": "user", "content": ROLE_SYS[role] + "\n\n" +
                     "NHIỆM VỤ: Đọc kỹ ngữ cảnh và câu trả lời, sau đó:\n"
                     "1. Chọn 1 trong 3 nhãn: no, intrinsic, extrinsic\n"
                     "2. Viết theo format: label: <nhãn>\n"
                     "3. Giải thích ngắn gọn lý do (1-2 câu)\n\n" + case}
                ]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(text, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs, 
                    max_new_tokens=64, 
                    do_sample=False, 
                    temperature=0.1,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id
                )
                dec = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                print(f"  {role}: {dec[:100]}...")  # Log để debug
                y = parse_label(dec)
                if y:
                    votes.append(y)

        maj = max(set(votes), key=votes.count) if votes else "no"

        judge_msgs = [
            {"role": "user", "content": JUDGE_SYS + "\n\n" +
             f"KẾT QUẢ VOTES:\n"
             f"- Literalist: {votes[0] if len(votes) > 0 else 'N/A'}\n"
             f"- Skeptic: {votes[1] if len(votes) > 1 else 'N/A'}\n"
             f"- Verifier: {votes[2] if len(votes) > 2 else 'N/A'}\n\n"
             f"QUYẾT ĐỊNH CUỐI CÙNG: Chỉ in 1 từ duy nhất: no hoặc intrinsic hoặc extrinsic."}
        ]
        jtext = tok.apply_chat_template(judge_msgs, tokenize=False, add_generation_prompt=True)
        jinputs = tok(jtext, return_tensors="pt").to(model.device)
        jout = model.generate(
            **jinputs, 
            max_new_tokens=3, 
            do_sample=False, 
            temperature=0.1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id
        )
        jdec = tok.decode(jout[0][jinputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"  judge: {jdec[:100]}...")  # Log để debug
        final = parse_label(jdec) or maj

        if final not in LABELS:
            final = "no"
        preds.append(final)

        print(f"  votes: {votes}, maj: {maj}, final: {final}")  # Log để debug
        if votes:
            pbar.set_postfix({"maj": maj, "votes": {l: votes.count(l) for l in set(votes)}})

    pd.DataFrame({"id": ids, "predict_label": preds}).to_csv(output_csv, index=False)
    print(f"✅ Wrote predictions to {output_csv} ({len(ids)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    debate_predict_csv(args.input_csv, args.output_csv, args.ckpt, rounds=args.rounds)


