from __future__ import annotations
import json, re, zipfile, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path('/home/tharun/Downloads')
OUT = ROOT / 'romanized_telugu_dataset_cleaned'
OUT.mkdir(exist_ok=True)

ARCHIVES = sorted(ROOT.glob('WhatsApp Chat*.zip'))
# Only the two friend groups are consented/usable for training.
ARCHIVE_FILTER = {'WhatsApp Chat - Bokkale.zip', 'WhatsApp Chat - fdfszfdfs.zip'}
ARCHIVES = [a for a in ARCHIVES if a.name in ARCHIVE_FILTER]
# Only use a likely self-name for role assignment; it is never written to output.
SELF_NAME_HINTS = {'tharun', 'tharun chougoni', 'chougoni'}

# WhatsApp export variants:
#   [12/08/2026, 10:53:00 PM] Name: text        (bracket form, seconds optional)
#   12/08/2026, 10:53 pm - Name: text           (dash form)
# The timestamp group greedily consumes optional seconds AND am/pm so a
# "10:53:00" colon can never be misread as the sender separator.
HEADER = re.compile(
    r'^(?:\[(\d{1,4}[/-]\d{1,2}[/-]\d{1,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)\]\s+'
    r'|(\d{1,4}[/-]\d{1,2}[/-]\d{1,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)\s*[-:]\s*)'
    r'([^:]+?):\s?(.*)$',
    re.I,
)
SYSTEM = re.compile(r'^\s*(?:messages and calls are end-to-end encrypted|you created this group|\+?\d[\d ()-]{7,}|this message was deleted|<media omitted>|image omitted|video omitted|audio omitted|sticker omitted|gif omitted|document omitted|contact card omitted|location omitted)\s*$', re.I)
URL = re.compile(r'\b(?:https?://|www\.)\S+|\b(?:wa\.me|t\.me|maps\.google\.)\S+', re.I)
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
PHONE = re.compile(r'(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)')
IP = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
UPI = re.compile(r'\b[\w.-]{2,}@[\w.-]{2,}\b', re.I)
SENSITIVE = re.compile(r'\b(?:otp|one[- ]time password|password|passcode|pin|cvv|aadhaar|aadhar|pan card|bank account|account number|ifsc|credit card|debit card|upi id|secret key|api key|token|medical report|prescription)\b', re.I)
NON_TEXT = re.compile(r'^\s*<(?:media|image|video|audio|sticker|document|contact|location)[^>]*>\s*$', re.I)
# System-event lines WhatsApp embeds in exports (sender-name independent).
SYSTEM_EVENT = re.compile(
    r'(?:security code|invite link|changed the (?:group|subject|description|icon)'
    r'|you (?:left|were removed|were added|joined)|removed you|added you'
    r'|(?:missed )?(?:voice|video) call|incoming call|call (?:ended|started)'
    r'|this message was edited|deleted this message|this message was deleted'
    r'|messages and calls are end-to-end encrypted|you created this group)',
    re.I,
)
# After redaction: a message that is ONLY placeholders (bare mentions, bare
# links, bare phone/email) carries no signal -> drop it.
PLACEHOLDER_ONLY = re.compile(r'^(?:<URL>|<EMAIL>|<PHONE>|<UPI>|@?<PERSON>|\s)+$')


def norm_name(name: str) -> str:
    # Strip Unicode direction marks ('\u200e\u200f') that WhatsApp export embeds
    # around names (e.g. '\u200eYou'), then normalize whitespace and case.
    return re.sub(r'\s+', ' ', name.replace('\u200e', '').replace('\u200f', '').strip()).casefold()


def anonymize_text(text: str, names: set[str]) -> tuple[str, bool]:
    # Strip bidirectional/isolate control marks (U+200E/200F/2068/2069) that
    # WhatsApp embeds around names and mentions.
    text = text.replace('\u200e', '').replace('\u200f', '').replace('\u2068', '').replace('\u2069', '').strip()
    if not text or SYSTEM.match(text) or NON_TEXT.match(text) or SYSTEM_EVENT.search(text):
        return '', True
    # Drop likely high-risk messages rather than pretending regex can safely sanitize them.
    if SENSITIVE.search(text):
        return '', True
    text = URL.sub('<URL>', text)
    text = EMAIL.sub('<EMAIL>', text)
    text = IP.sub('<IP>', text)
    text = PHONE.sub('<PHONE>', text)
    text = UPI.sub('<UPI>', text)
    # Replace known sender/contact names in one regex pass; a per-name pass is
    # prohibitively slow for large group exports.
    if names:
        parts = [re.escape(n) for n in sorted(names, key=len, reverse=True) if len(n) >= 3]
        if parts:
            text = re.sub(r'(?<!\w)(?:' + '|'.join(parts) + r')(?!\w)', '<PERSON>', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove obvious export artifacts and very short noise.
    text = re.sub(r'\s*\(file attached\)\s*', ' ', text, flags=re.I)
    # After redaction: bare mentions / bare links carry no content.
    if PLACEHOLDER_ONLY.match(text):
        return '', True
    if len(text) < 2 or not re.search(r'[A-Za-z\u0C00-\u0C7F]', text):
        return '', True
    return text, False


def extract_texts(zpath: Path):
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.txt'):
                continue
            raw = z.read(info).decode('utf-8-sig', errors='replace')
            yield info.filename, raw


def parse(raw: str):
    current = None
    for line in raw.splitlines():
        # WhatsApp prefixes some export lines with a left-to-right mark.
        line = line.lstrip('\u200e\u200f')
        m = HEADER.match(line)
        if m:
            if current: yield current
            g = m.groups()
            date = g[0] or g[2]
            tm = g[1] or g[3]
            sender, body = g[4], g[5]
            current = {'date': date, 'time': tm, 'sender': sender.strip(), 'text': body}
        elif current and line.strip():
            current['text'] += '\n' + line.strip()
    if current: yield current

all_rows = []
report = {'archives': {}, 'messages_seen': 0, 'messages_kept': 0, 'messages_dropped': 0, 'messages_redacted': 0, 'notes': []}
all_names = set()
raw_rows = []

for zpath in ARCHIVES:
    seen = kept = dropped = 0
    for fname, raw in extract_texts(zpath):
        for row in parse(raw):
            seen += 1; report['messages_seen'] += 1
            all_names.add(norm_name(row['sender']))
            raw_rows.append((zpath.name, fname, row))
    report['archives'][zpath.name] = {'messages_seen': seen}

# Include names found in VCF files, but only as an in-memory redaction dictionary.
for zpath in ARCHIVES:
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.filename.lower().endswith('.vcf'):
                txt = z.read(info).decode('utf-8', errors='ignore')
                for nm in re.findall(r'(?im)^FN[^:]*:(.+)$', txt):
                    all_names.add(norm_name(nm))

# Assign stable anonymous speaker IDs per archive, never based on original names.
speaker_maps = {}
for archive, fname, row in raw_rows:
    speaker_maps.setdefault(archive, {})
    key = norm_name(row['sender'])
    if key not in speaker_maps[archive]:
        speaker_maps[archive][key] = f'speaker_{len(speaker_maps[archive])}'

clean = []
by_conv = defaultdict(list)
for archive, fname, row in raw_rows:
    text, dropped_flag = anonymize_text(row['text'], all_names)
    if dropped_flag:
        report['messages_dropped'] += 1
        continue
    sp = speaker_maps[archive][norm_name(row['sender'])]
    # Hash archive + source filename + date/time; no original text or names in ID.
    cid = hashlib.sha256(f'{archive}|{fname}'.encode()).hexdigest()[:16]
    rec = {'conversation_id': cid, 'source_archive': hashlib.sha256(archive.encode()).hexdigest()[:12], 'speaker': sp, 'text': text}
    clean.append(rec); by_conv[cid].append(rec)
    report['messages_kept'] += 1
    report['archives'][archive]['messages_kept'] = report['archives'][archive].get('messages_kept', 0) + 1

# Save anonymous turn format. This is the source-of-truth and is safe to inspect.
with (OUT/'cleaned_turns.jsonl').open('w', encoding='utf-8') as f:
    for r in clean:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# Make reviewable conversation records. Roles remain speaker_0/speaker_1 intentionally:
# assigning assistant/user automatically would risk training the wrong person.
with (OUT/'cleaned_conversations.jsonl').open('w', encoding='utf-8') as f:
    for cid, turns in by_conv.items():
        if len(turns) < 2: continue
        f.write(json.dumps({'conversation_id': cid, 'messages': [{'role': x['speaker'], 'content': x['text']} for x in turns]}, ensure_ascii=False) + '\n')

# Candidate SFT file only when the export contains a sender matching the self-name hint.
# Otherwise leave it empty and explain why, rather than inventing roles.
with (OUT/'sft_candidates.jsonl').open('w', encoding='utf-8') as f:
    for cid, turns in by_conv.items():
        speakers = {x['speaker'] for x in turns}
        if len(speakers) != 2: continue
        # We cannot safely map anonymous IDs back to names after processing, so require manual mapping.
        continue

report['notes'] += [
    'All original names, archive names, contact-card names, phone numbers, URLs, emails, IPs and UPI-like identifiers are excluded from output records.',
    'Messages containing high-risk secrets/financial/medical keywords or media placeholders were dropped.',
    'Roles are intentionally anonymous speaker_0/speaker_1; do not train directly until the assistant speaker is identified and mapped.',
    'Original ZIP archives were not modified.',
]
(OUT/'processing_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'output_dir': str(OUT), 'archives': len(ARCHIVES), 'messages_seen': report['messages_seen'], 'messages_kept': report['messages_kept'], 'messages_dropped': report['messages_dropped']}, indent=2))