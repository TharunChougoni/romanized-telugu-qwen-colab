# Romanized Telugu Qwen SFT — Colab project

This project prepares privacy-filtered WhatsApp exports for supervised fine-tuning of `Qwen/Qwen3-4B-Instruct-2507`.

## Recommended pipeline

```text
cleaned anonymous conversations
→ review speaker mapping
→ build prompt/completion examples
→ validate Qwen/TRL formatting
→ QLoRA with assistant-only loss
→ evaluate against the base model
```

The builder intentionally preserves edgy style: profanity, insults, sarcasm, sexual language, taboo wording, and other “unhinged” tone are not filtered at the SFT-builder stage. The earlier cleaner may already have removed direct identifiers, secrets, media placeholders, and messages matching high-risk financial/medical keywords; those cannot be restored from the cleaned file.

## Colab

Open the notebook:

https://colab.research.google.com/github/TharunChougoni/romanized-telugu-qwen-colab/blob/main/notebooks/prepare_qwen_dataset.ipynb

Private data should live in Drive:

```text
/content/drive/MyDrive/romanized_telugu_dataset_cleaned/
```

Required private files:

- `cleaned_conversations.jsonl`
- `speaker_mapping.json` after manual review

The notebook clones the public code repository and writes generated data back to Drive.

## Output format

The builder writes conversational prompt-completion records:

```json
{"prompt":[{"role":"user","content":"..."}],"completion":[{"role":"assistant","content":"..."}]}
```

This is the format used by current TRL conversational SFT workflows. `assistant_only_loss=True` trains only on the completion/assistant response. Qwen3 is among the model families for which TRL supports assistant-generation masks.

## Important speaker rule

Anonymous speaker IDs can vary between conversations. Do not assume `speaker_0` is you globally. Map each `conversation_id` to the speaker ID representing you, then the builder maps your messages to `assistant` and everyone else to `user`.

## Privacy

Keep the repository private if you add any derived conversation data. Never commit raw WhatsApp ZIPs, contact cards, `cleaned_conversations.jsonl`, `speaker_mapping.json`, generated SFT files, secrets, or adapters.

## Public code files

- `notebooks/prepare_qwen_dataset.ipynb`
- `src/build_lora_dataset.py`
- `src/prepare_qwen_colab.py`
- `requirements-colab.txt`

The repository contains code only; private chat data remains in Drive.

## Research basis

The pipeline follows the current Hugging Face TRL SFT guidance: conversational prompt-completion data, chat-template processing, conversation-level splits, packing only after validation, PEFT/LoRA adapters, and assistant-only loss. Do not manually flatten messages into token IDs before training.
