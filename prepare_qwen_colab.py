"""Prepare cleaned WhatsApp conversations for Qwen3-4B-Instruct-2507 SFT in Colab.

Usage in Colab:
  !pip install -U transformers datasets
  !python prepare_qwen_colab.py \
      --input cleaned_conversations.jsonl \
      --output qwen_sft \
      --assistant-speaker speaker_0

IMPORTANT: choose the anonymous speaker ID that represents YOU. If it differs
between archives/conversations, use --mapping-json instead.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from collections import Counter

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def load_mapping(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Format: {"conversation_id": "speaker_1"}
    if not isinstance(data, dict):
        raise ValueError("mapping JSON must be {conversation_id: assistant_speaker}")
    return {str(k): str(v) for k, v in data.items()}


def normalize_messages(raw_messages, assistant_speaker: str):
    out = []
    for m in raw_messages:
        speaker = str(m.get("role", ""))
        text = str(m.get("content", "")).strip()
        if not text:
            continue
        role = "assistant" if speaker == assistant_speaker else "user"
        # Merge adjacent messages from the same effective role.
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + text
        else:
            out.append({"role": role, "content": text})
    # SFT needs both sides and an assistant target.
    if not any(x["role"] == "assistant" for x in out):
        return None
    if not any(x["role"] == "user" for x in out):
        return None
    # Start at a user turn; discard leading assistant text.
    while out and out[0]["role"] != "user":
        out.pop(0)
    if len(out) < 2:
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--assistant-speaker", default="speaker_0")
    ap.add_argument("--mapping-json")
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-examples", type=int, default=0)
    args = ap.parse_args()

    # Import only when run in Colab/local environment.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    mapping = load_mapping(args.mapping_json)
    examples = []
    input_path = Path(args.input)
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = str(rec.get("conversation_id", ""))
        speaker = mapping.get(cid, args.assistant_speaker)
        msgs = normalize_messages(rec.get("messages", []), speaker)
        if msgs is None:
            continue
        # Validate against the actual Qwen chat template before writing.
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        if not rendered.strip():
            continue
        examples.append({"conversation_id": cid, "messages": msgs})

    # Conversation-level split prevents near-duplicate turns leaking across sets.
    random.Random(args.seed).shuffle(examples)
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    n_val = max(1, round(len(examples) * args.validation_fraction)) if len(examples) >= 10 else 0
    val = examples[:n_val]
    train = examples[n_val:]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("validation.jsonl", val)]:
        with (out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    info = {
        "base_model": MODEL,
        "format": "Qwen chat messages; rendered with tokenizer.apply_chat_template",
        "thinking": False,
        "train_examples": len(train),
        "validation_examples": len(val),
        "assistant_speaker_default": args.assistant_speaker,
        "mapping_json": args.mapping_json,
        "note": "Review samples and verify assistant speaker mapping before training.",
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print("Example rendered prompt:\n", tokenizer.apply_chat_template(train[0]["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)[:1200] if train else "none")


if __name__ == "__main__":
    main()
