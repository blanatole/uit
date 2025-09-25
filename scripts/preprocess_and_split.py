import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import unicodedata
import re


class VietnameseTextPreprocessor:
    vowel_map = [
        ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
    ]

    tone_map = ['', 'f', 's', 'r', 'x', 'j']
    vowel_to_ids = {}

    special_map = {
        "nàozz?": "nào?",
        "nao?": "nào?",
        "nao??": "nào?",
        "nao???": "nào?",
        "nàoa?": "nào?",
        "bao nhieu?": "bao nhiêu?",
        "gi?": "gì?",
        "gìy?": "gì?",
        "gìi?": "gì?",
        "ạh?": "ạ?",
        "nàoa": "nào",
        "nàoz": "nào",
        "nàozz": "nào",
        "nàoh": "nào",
        "nàoe": "nào",
        "nàot": "nào",
        "nàoo": "nào",
    }

    @classmethod
    def initialize_vowel_to_ids(cls):
        for i in range(len(cls.vowel_map)):
            for j in range(len(cls.vowel_map[i]) - 1):
                cls.vowel_to_ids[cls.vowel_map[i][j]] = (i, j)

    @staticmethod
    def unicode_normalize(text):
        return unicodedata.normalize('NFC', text)

    @classmethod
    def is_valid_vietnamese_word(cls, word):
        chars = list(word)
        vowel_index = -1
        for index, char in enumerate(chars):
            x, y = cls.vowel_to_ids.get(char, (-1, -1))
            if x != -1:
                if vowel_index == -1:
                    vowel_index = index
                else:
                    if index - vowel_index != 1:
                        return False
                    vowel_index = index
        return True

    @classmethod
    def standardize_vietnamese_tone(cls, word):
        if not cls.is_valid_vietnamese_word(word):
            return word

        chars = list(word)
        tone = 0
        vowel_indices = []
        is_qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = cls.vowel_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9 and index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                is_qu_or_gi = True
            elif x == 5 and index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                is_qu_or_gi = True
            if y != 0:
                tone = y
                chars[index] = cls.vowel_map[x][0]
            if not is_qu_or_gi or index != 1:
                vowel_indices.append(index)

        if len(vowel_indices) < 2:
            if is_qu_or_gi:
                if len(chars) == 2:
                    x, y = cls.vowel_to_ids.get(chars[1])
                    chars[1] = cls.vowel_map[x][tone]
                else:
                    x, y = cls.vowel_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = cls.vowel_map[x][tone]
                    else:
                        chars[1] = cls.vowel_map[5][tone] if chars[1] == 'i' else cls.vowel_map[9][tone]
                return ''.join(chars)
            return word

        for index in vowel_indices:
            x, y = cls.vowel_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = cls.vowel_map[x][tone]
                return ''.join(chars)

        if len(vowel_indices) == 2:
            if vowel_indices[-1] == len(chars) - 1:
                x, y = cls.vowel_to_ids[chars[vowel_indices[0]]]
                chars[vowel_indices[0]] = cls.vowel_map[x][tone]
            else:
                x, y = cls.vowel_to_ids[chars[vowel_indices[1]]]
                chars[vowel_indices[1]] = cls.vowel_map[x][tone]
        else:
            x, y = cls.vowel_to_ids[chars[vowel_indices[1]]]
            chars[vowel_indices[1]] = cls.vowel_map[x][tone]
        return ''.join(chars)

    @classmethod
    def standardize_sentence_tone(cls, sentence):
        words = sentence.split()
        for index, word in enumerate(words):
            if not word:
                return " "
            sep = "\x01"
            marked = re.sub(r'(^\W*)([\w.]*\w+)(\W*$)', rf'\1{sep}\2{sep}\3', word)
            parts = marked.split(sep)
            if len(parts) == 3:
                original_core = parts[1]
                lower_core = original_core.lower()
                standardized_lower = cls.standardize_vietnamese_tone(lower_core)

                if len(standardized_lower) == len(original_core):
                    rebuilt = []
                    for i, ch in enumerate(standardized_lower):
                        rebuilt.append(ch.upper() if original_core[i].isupper() else ch)
                    parts[1] = ''.join(rebuilt)
                else:
                    if original_core.isupper():
                        parts[1] = standardized_lower.upper()
                    elif original_core[:1].isupper() and original_core[1:].islower():
                        parts[1] = standardized_lower.capitalize()
                    else:
                        parts[1] = standardized_lower
            else:
                parts = [marked]
            words[index] = ''.join(parts)
        return ' '.join(words)

    @classmethod
    def fix_repeated_chars(cls, sentence):
        if not sentence:
            return sentence

        def is_vietnamese_like(token: str) -> bool:
            if not token:
                return False
            if any(ch in 'đĐ' for ch in token):
                return True
            return any(ord(ch) > 127 for ch in token)

        def collapse_lower_repeats(token: str) -> str:
            out = []
            prev = None
            run = 0
            for ch in token:
                if ch == prev:
                    run += 1
                else:
                    if prev is not None:
                        out.extend([prev] if (run >= 2 and prev.isalpha() and prev.islower()) else [prev] * run)
                    prev = ch
                    run = 1
            if prev is not None:
                out.extend([prev] if (run >= 2 and prev.isalpha() and prev.islower()) else [prev] * run)
            return ''.join(out)

        chunks = []
        current = []
        in_letters = None
        for ch in sentence:
            is_letter = ch.isalpha()
            if in_letters is None:
                in_letters = is_letter
                current.append(ch)
            elif is_letter == in_letters:
                current.append(ch)
            else:
                chunk = ''.join(current)
                if in_letters and is_vietnamese_like(chunk):
                    chunks.append(collapse_lower_repeats(chunk))
                else:
                    chunks.append(chunk)
                current = [ch]
                in_letters = is_letter

        if current:
            chunk = ''.join(current)
            if in_letters and is_vietnamese_like(chunk):
                chunks.append(collapse_lower_repeats(chunk))
            else:
                chunks.append(chunk)

        return ''.join(chunks)

    @classmethod
    def preprocess(cls, text):
        if text:
            for src, dst in cls.special_map.items():
                text = text.replace(src, dst)
        text = cls.fix_repeated_chars(text)
        text = cls.standardize_sentence_tone(text)
        if text:
            text = re.sub(r"\?{2,}", "?", text)
        return text


class CSVTextProcessor:
    def __init__(self, input_csv, output_csv=None, log_txt=None, preprocessor=None):
        self.input_csv = Path(input_csv)
        if output_csv is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_csv = self.input_csv.with_name(f"{self.input_csv.stem}_{date_str}.csv")
        if log_txt is None:
            log_txt = self.input_csv.with_name(f"{self.input_csv.stem}_log.txt")
        self.output_csv = Path(output_csv)
        self.log_txt = Path(log_txt)
        self.preprocessor = preprocessor or VietnameseTextPreprocessor
        if not getattr(self.preprocessor, "vowel_to_ids", None):
            self.preprocessor.initialize_vowel_to_ids()

    def _strip_marker(self, text):
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        for phrase in ["Adversarial Question:", "Question:", 'có thể như sau:']:
            idx = text.find(phrase)
            if idx != -1:
                before = text[:idx]
                after = text[idx + len(phrase):].lstrip()
                tokens = self._extract_slash_tokens(before)
                if tokens:
                    preserved = []
                    seen = set()
                    for t in tokens:
                        if t not in seen:
                            seen.add(t)
                            preserved.append(t)
                    to_prepend = [t for t in preserved if t not in after]
                    if to_prepend:
                        after = (" ".join(to_prepend) + " " + after).strip()
                return after
        return text

    def _extract_slash_tokens(self, text):
        pattern = r"(?:(?<=^)|(?<=\s))(?:[\w]+(?:/[\w]+)+(?:-[\w]+)*)"
        return [m.group(0) for m in re.finditer(pattern, text)]

    def _clean_text(self, text):
        stripped = self._strip_marker(text)
        cleaned = self.preprocessor.preprocess(stripped) if stripped else stripped
        return cleaned.strip()

    def _clean_prompt(self, text):
        cleaned = self._clean_text(text)
        return cleaned.replace('@', '') if cleaned else cleaned

    def process(self):
        df = pd.read_csv(self.input_csv, dtype=str, keep_default_na=False)
        required_cols = ["id", "context", "prompt", "response"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        logs = []

        def _apply_row(row):
            original_prompt = row.get("prompt", "")
            original_response = row.get("response", "")

            cleaned_prompt = self._clean_prompt(original_prompt)
            cleaned_response = self._clean_text(original_response)

            if cleaned_prompt != original_prompt:
                logs.append({
                    "id": row.get("id", ""),
                    "field": "prompt",
                    "before": original_prompt,
                    "after": cleaned_prompt,
                })
            if cleaned_response != original_response:
                logs.append({
                    "id": row.get("id", ""),
                    "field": "response",
                    "before": original_response,
                    "after": cleaned_response,
                })

            row["prompt"] = cleaned_prompt
            row["response"] = cleaned_response
            return row

        df = df.apply(_apply_row, axis=1)

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False, encoding="utf-8")

        self.log_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_txt, "w", encoding="utf-8") as f:
            if logs:
                for entry in logs:
                    before = (entry.get("before", "") or "").replace("\n", "\\n")
                    after = (entry.get("after", "") or "").replace("\n", "\\n")
                    f.write(f"ID={entry.get('id', '')} {entry.get('field', '').upper()}:\n")
                    f.write(f"Before: {before}\n")
                    f.write(f"After:  {after}\n")
                    f.write("---\n")
            else:
                f.write("No changes\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese text and stratified split")
    parser.add_argument("--input", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "vihallu-train.csv"))
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction for temp (val+test)")
    parser.add_argument("--val_size", type=float, default=0.5, help="Fraction of temp to use as test (val/test split)")
    args = parser.parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Clean text columns and write a cleaned CSV + log
    date_str = datetime.now().strftime("%Y%m%d")
    cleaned_csv = outdir / f"{input_csv.stem}_cleaned_{date_str}.csv"
    log_txt = outdir / f"{input_csv.stem}_clean_log_{date_str}.txt"
    processor = CSVTextProcessor(input_csv=input_csv, output_csv=cleaned_csv, log_txt=log_txt)
    processor.process()

    # 2) Load cleaned CSV and normalize labels
    df = pd.read_csv(cleaned_csv, dtype=str, keep_default_na=False)
    df = df.dropna(subset=["context", "prompt", "response", "label"]).copy()
    df["label"] = df["label"].str.strip().str.lower()

    # filter to allowed labels
    allowed = {"no", "intrinsic", "extrinsic"}
    df = df[df["label"].isin(allowed)].reset_index(drop=True)

    # 3) Stratified split 80/10/10
    train_df, temp_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=args.random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=args.val_size, stratify=temp_df["label"], random_state=args.random_state
    )

    # 4) Write jsonl
    (outdir / "train.jsonl").write_text("\n".join(train_df.to_json(orient="records", lines=True, force_ascii=False).splitlines()), encoding="utf-8")
    (outdir / "val.jsonl").write_text("\n".join(val_df.to_json(orient="records", lines=True, force_ascii=False).splitlines()), encoding="utf-8")
    (outdir / "test.jsonl").write_text("\n".join(test_df.to_json(orient="records", lines=True, force_ascii=False).splitlines()), encoding="utf-8")

    # 5) Save small label stats
    stats = {
        "counts": df["label"].value_counts().to_dict(),
        "train_counts": train_df["label"].value_counts().to_dict(),
        "val_counts": val_df["label"].value_counts().to_dict(),
        "test_counts": test_df["label"].value_counts().to_dict(),
    }
    (outdir / f"split_stats_{date_str}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Cleaned CSV: {cleaned_csv}")
    print(f"Log file:    {log_txt}")
    print(f"Wrote: {outdir / 'train.jsonl'}, {outdir / 'val.jsonl'}, {outdir / 'test.jsonl'}")


if __name__ == "__main__":
    main()


