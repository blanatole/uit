import re, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["no","intrinsic","extrinsic"]
PAT = re.compile(r"\b(no|intrinsic|extrinsic)\b", re.I)

ROLE_SYS = {
"literalist": "Vai trò: BÁM VĂN BẢN. Chỉ quyết định dựa trên NGỮ CẢNH, tránh suy diễn.",
"skeptic":    "Vai trò: HOÀI NGHI. Chủ động tìm lỗi, mâu thuẫn, thiếu chứng cứ.",
"verifier":   "Vai trò: THẨM ĐỊNH. Đối chiếu từng mệnh đề với NGỮ CẢNH, ghi rõ có/không chứng cứ."
}

JUDGE_SYS = (
"Bạn là TRỌNG TÀI. Nhận 3 câu trả lời (nhãn + lý do) từ các vai trò.\n"
"Quy tắc: ƯU TIÊN BẰNG CHỨNG trong NGỮ CẢNH. Nếu có mâu thuẫn rõ → intrinsic. "
"Nếu thông tin không có trong ngữ cảnh → extrinsic. Nếu đầy đủ chứng cứ → no.\n"
"Chỉ in nhãn cuối cùng (no|intrinsic|extrinsic)."
)

def build_case(row):
    return (
        f"NGỮ CẢNH:\n{row['context']}\n\n"
        f"CÂU HỎI:\n{row['prompt']}\n\n"
        f"CÂU TRẢ LỜI CỦA TRỢ LÝ:\n{row['response']}\n"
    )

def agent_prompt(role_text, case_text):
    return (
        role_text + "\n\n"
        "Nhiệm vụ: Trả về đúng định dạng: `label: <no|intrinsic|extrinsic>` và 1 câu lý do.\n\n"
        + case_text
    )

def parse_label(text):
    m = PAT.search(text)
    return m.group(1).lower() if m else None

@torch.inference_mode()
def debate_predict(df_path, model_dir="out/gemma-vihallu/adapter", rounds=1):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)

    df = pd.read_json(df_path, lines=True)
    preds = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="Debate", dynamic_ncols=True)
    for _, row in pbar:
        case = build_case(row)
        votes = []
        for _ in range(rounds):
            for role in ["literalist","skeptic","verifier"]:
                # Gemma chat template không hỗ trợ role "system" → gộp vào user prompt
                msgs = [
                    {"role":"user","content":agent_prompt(ROLE_SYS[role], case)}
                ]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(text, return_tensors="pt").to(model.device)
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False, eos_token_id=tok.eos_token_id)
                dec = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                y = parse_label(dec)
                if y: votes.append(y)

        if votes:
            maj = max(set(votes), key=votes.count)
        else:
            maj = "no"

        judge_msgs = [
            {"role":"user","content":JUDGE_SYS + "\n\n" + f"{case}\n\nCác câu trả lời từ 3 vai trò (gộp nhiều vòng nếu có):\n{votes}\n\nHãy in nhãn cuối (no|intrinsic|extrinsic)."}
        ]
        jtext = tok.apply_chat_template(judge_msgs, tokenize=False, add_generation_prompt=True)
        jinputs = tok(jtext, return_tensors="pt").to(model.device)
        jout = model.generate(**jinputs, max_new_tokens=4, do_sample=False, eos_token_id=tok.eos_token_id)
        jdec = tok.decode(jout[0][jinputs["input_ids"].shape[1]:], skip_special_tokens=True)
        final = parse_label(jdec) or maj
        preds.append(final)
        # Cập nhật tiến trình
        if votes:
            pbar.set_postfix({"maj": maj, "votes": {l: votes.count(l) for l in set(votes)}})
    return preds
