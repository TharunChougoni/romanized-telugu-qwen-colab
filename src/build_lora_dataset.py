"""Build LoRA-ready Qwen conversational SFT data.

Input: cleaned_conversations.jsonl with anonymous speaker roles.
Output: prompt-completion JSONL, where completion is one target speaker turn.
Use --all-speakers when consent permits learning from every participant.

This preserves offensive, profane, sexual, sarcastic, and otherwise edgy style
content. It still relies on the earlier cleaner for direct identifiers/secrets;
those cannot be restored from cleaned_conversations.jsonl.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

def read_mapping(path):
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("mapping must be {conversation_id: assistant_speaker}")
    return {str(k): str(v) for k, v in d.items()}

def clean_turns(raw):
    out=[]
    for m in raw:
        speaker=str(m.get("role", "")); text=str(m.get("content", "")).strip()
        if not speaker or not text: continue
        # Preserve wording exactly; only merge adjacent turns from same speaker.
        if out and out[-1]["speaker"] == speaker:
            out[-1]["text"] += "\n" + text
        else:
            out.append({"speaker":speaker,"text":text})
    return out

def make_examples(rec, assistant_speaker, max_context_turns, min_chars):
    turns=clean_turns(rec.get("messages", [])); examples=[]
    for i,t in enumerate(turns):
        if t["speaker"] != assistant_speaker or len(t["text"]) < min_chars: continue
        start=max(0, i-max_context_turns)
        context=turns[start:i]
        while context and context[0]["speaker"] == assistant_speaker: context.pop(0)
        if not context or not any(x["speaker"] != assistant_speaker for x in context): continue
        prompt=[]
        for x in context:
            prompt.append({"role":"assistant" if x["speaker"]==assistant_speaker else "user", "content":x["text"]})
        examples.append({"conversation_id":str(rec.get("conversation_id","")),"prompt":prompt,"completion":[{"role":"assistant","content":t["text"]}]})
    return examples

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--mapping-json")
    ap.add_argument("--all-speakers",action="store_true",help="Use every participant as an assistant target in separate examples")
    ap.add_argument("--max-context-turns",type=int,default=8)
    ap.add_argument("--min-assistant-chars",type=int,default=2)
    ap.add_argument("--validation-fraction",type=float,default=.1)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    mapping=read_mapping(args.mapping_json); all_examples=[]; conversations=[]
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec=json.loads(line); cid=str(rec.get("conversation_id",""))
            turns=clean_turns(rec.get("messages", []))
            speakers=sorted({x["speaker"] for x in turns})
            targets=speakers if args.all_speakers else ([mapping[cid]] if cid in mapping else [])
            for speaker in targets:
                conversations.append(cid)
                all_examples.extend(make_examples(rec,speaker,args.max_context_turns,args.min_assistant_chars))

    # Deduplicate exact prompt/completion pairs while preserving the first copy.
    unique=[]; seen=set()
    for x in all_examples:
        key=json.dumps(x,ensure_ascii=False,sort_keys=True)
        if key not in seen:
            seen.add(key); unique.append(x)
    all_examples=unique
    # Split by conversation ID to prevent near-duplicate context leakage.
    rng=random.Random(args.seed)
    ids=sorted(set(x["conversation_id"] for x in all_examples))
    rng.shuffle(ids)
    if len(ids) > 1:
        n=max(1, round(len(ids)*args.validation_fraction))
        n=min(n, len(ids)-1)
        val_ids=set(ids[:n])
        train=[x for x in all_examples if x["conversation_id"] not in val_ids]
        val=[x for x in all_examples if x["conversation_id"] in val_ids]
    elif len(all_examples) > 1:
        # Fallback for a single conversation: avoid an empty validation file.
        rng.shuffle(all_examples)
        n=max(1, round(len(all_examples)*args.validation_fraction))
        n=min(n, len(all_examples)-1)
        val=all_examples[:n]
        train=all_examples[n:]
    else:
        train=[]
        val=[]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("train.jsonl",train),("validation.jsonl",val)):
        with (out/name).open("w",encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps({"prompt":x["prompt"],"completion":x["completion"]},ensure_ascii=False)+"\n")
    info={"base_model":MODEL,"format":"conversational prompt-completion","train_examples":len(train),"validation_examples":len(val),"mapped_conversations":len(ids),"preserves_edgy_content":True,"note":"Only direct identifiers/secrets removed by the earlier cleaner are absent; no additional sensitivity filtering is done here."}
    (out/"dataset_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2))
if __name__=="__main__": main()
