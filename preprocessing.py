import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab', quiet=True)

def clean_fmp_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def split_transcript(text: str) -> dict:
    """
    Splits prepared remarks from Q&A at the moderator's handoff to the first analyst
    question (e.g. "...take our first question from Erik Woodring...").
    """
    handoff_markers = ["first question", "first caller"]

    text_lower = text.lower()
    split_idx = None
    for marker in handoff_markers:
        idx = text_lower.find(marker)
        if idx != -1 and (split_idx is None or idx < split_idx):
            split_idx = idx

    if split_idx is None:
        # Fallback for calls that skip straight to naming the first analyst: the
        # moderator's *second* handoff still lands squarely inside the Q&A section.
        idx = text_lower.find("next question")
        split_idx = idx if idx != -1 else len(text)  # last resort: no Q&A found

    return {
        "prepared": text[:split_idx],
        "qa": text[split_idx:],
    }

_FMP_TURN_RE = re.compile(r"^([A-Za-z][\w.'-]*(?:\s+[A-Za-z][\w.'-]*){0,3}):\s*(.+)$")

def segment_qa_pairs(qa_text: str) -> list:
    # Splits the Q&A half (split_transcript()["qa"]) into [{"question", "answer"}, ...].

    pairs = []
    pending_q, pending_a = [], []
    analyst = None  # name identified as the active slot's question-asker

    for raw_line in qa_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _FMP_TURN_RE.match(line)
        if not match:
            continue
        name, text = match.group(1).strip(), match.group(2).strip()

        if name.lower() == "operator":
            if pending_q and pending_a:
                pairs.append({"question": " ".join(pending_q).strip(),
                              "answer": " ".join(pending_a).strip()})
                pending_q, pending_a = [], []
            analyst = None
            continue

        if analyst is None or name == analyst:
            analyst = name
            if pending_a:
                pairs.append({"question": " ".join(pending_q).strip(),
                              "answer": " ".join(pending_a).strip()})
                pending_q, pending_a = [], []
            pending_q.append(text)
        else:
            pending_a.append(text)

    if pending_q and pending_a:
        pairs.append({"question": " ".join(pending_q).strip(),
                      "answer": " ".join(pending_a).strip()})
    return pairs

def clean_text(text: str) -> str:
    # Remove first 3 lines (credits and metadata)
    text = "\n".join(text.splitlines()[3:])
    # Remove speaker labels like "John Smith -- CFO"
    text = re.sub(r'^[A-Z][a-z]+ [A-Z][a-z]+ -- [\w\s]+$', '', text, flags=re.MULTILINE)
    # Remove operator lines
    text = re.sub(r'Operator\n.*?\n', '', text, flags=re.DOTALL)
    # Remove forward-looking statement boilerplate
    text = re.sub(r'This .*?safe harbor.*?\.', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sentence_tokenize(text: str) -> list:
    # strips whitespace and tokenizes sentences
    # removes sentences with 4 or fewer words (most likely headers)
    return [s.strip() for s in sent_tokenize(text) if len(s.split()) > 4]