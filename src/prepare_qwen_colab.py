from __future__ import annotations
import argparse, json, random
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

def load_mapping(path):
    if not path: return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict): raise ValueError("mapping must be {conversation_id: speaker_id}")
    return {str(k): str(v) for k, v in data.items()}

def normalize_messages(raw, assistant_speaker):
    out=[]
    for item in raw:
        speaker=str(item.get("role", "")); text=str(item.get("content", "")).strip()
        if not text: continue
        role="assistant" if speaker == assistant_speaker else "user"
        if out and out[-1]["role"] == role: out[-1]["content"] += "\n" + text
        else: out.append({"role": role, "content": text})
    while out and out[0]["role"] != "user": out.pop(0)
    if len(out)<2 or not any(x["role"]=="assistant" for x in out): return None
    if not any(x["role"]=="user" for x in out): return None
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--assistant-speaker", default="speaker_0"); ap.add_argument("--mapping-json")
    ap.add_argument("--validation-fraction", type=float, default=.1); ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    mapping=load_mapping(args.mapping_json); rows=[]
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        rec=json.loads(line); cid=str(rec.get("conversation_id", ""))
        msgs=normalize_messages(rec.get("messages", []), mapping.get(cid, args.assistant_speaker))
        if msgs:
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            rows.append({"messages":msgs})
    random.Random(args.seed).shuffle(rows)
    n=max(1, round(len(rows)*args.validation_fraction)) if len(rows)>=10 else 0
    val,train=rows[:n],rows[n:]
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name,data in (("train.jsonl",train),("validation.jsonl",val)):
        with (out/name).open("w",encoding="utf-8") as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    info={"base_model":MODEL,"train_examples":len(train),"validation_examples":len(val),"thinking":False}
    (out/"dataset_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2))

if __name__ == "__main__": main()

# In Colab, call main through a shell cell rather than importing this file.
# Example:
# !python src/prepare_qwen_colab.py --input cleaned_conversations.jsonl --output qwen_sft --mapping-json speaker_mapping.json

# SFT template (run in a separate notebook cell after reviewing qwen_sft/train.jsonl):
# from datasets import load_dataset
# from transformers import AutoTokenizer
# from trl import SFTConfig, SFTTrainer
# from peft import LoraConfig
# ds = load_dataset("json", data_files={"train":"qwen_sft/train.jsonl", "validation":"qwen_sft/validation.jsonl"})
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# def render(x): return {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)}
# ds = ds.map(render)
# trainer = SFTTrainer(model=MODEL, train_dataset=ds["train"], eval_dataset=ds["validation"], args=SFTConfig(output_dir="qwen3-romanized-telugu-lora", max_seq_length=2048, packing=True), peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules="all-linear", task_type="CAUSAL_LM"))
# trainer.train()
# trainer.save_model("qwen3-romanized-telugu-lora")
# tokenizer.save_pretrained("qwen3-romanized-telugu-lora")
