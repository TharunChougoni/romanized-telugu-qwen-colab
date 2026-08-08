# Romanized Telugu Qwen SFT — Colab project

This project prepares privacy-filtered WhatsApp exports for supervised fine-tuning of `Qwen/Qwen3-4B-Instruct-2507`.

## Colab workflow

1. Connect Colab to GitHub or upload/clone this repository.
2. Open `notebooks/prepare_qwen_dataset.ipynb`.
3. Run the cells in order.
4. Review anonymous conversations and create `speaker_mapping.json` before conversion.
5. Convert to Qwen `messages` format.
6. Run the optional QLoRA training section only after reviewing samples.

The repository deliberately does **not** include original WhatsApp exports or unreviewed private messages. Keep those local/private.

## Expected final format

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Important privacy step

The cleaner uses anonymous `speaker_0`, `speaker_1`, etc. IDs. These IDs may differ by archive/conversation. Do not assume one global speaker ID is you. Review samples and create a mapping before training.

## GitHub from Colab

You can connect Colab to GitHub through **File → Save a copy in GitHub**, or clone/push with a GitHub token. Do not commit WhatsApp ZIPs, raw exports, contact cards, or secrets.

## Files

- `notebooks/prepare_qwen_dataset.ipynb` — main Colab notebook
- `src/prepare_qwen_colab.py` — reusable converter
- `requirements-colab.txt` — Colab dependencies
- `cleaned_conversations.jsonl` — intermediate anonymized data; keep repository private
- `processing_report.json` — counts and limitations

The notebook defaults to conversion only. Training is included as a commented/template section so you can review the dataset first.
