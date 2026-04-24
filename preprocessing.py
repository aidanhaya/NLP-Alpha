import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab', quiet=True)

def split_transcript(text: str) -> dict:
    # Each marker is paired with a list of exclusion phrases.
    # If any exclusion phrase surrounds the match, it's skipped.
    qa_markers = [
        "question-and-answer",
        "questions and answers",
        "q&a",
        "question and answer session",
        "operator instructions",
        "we will now begin",
    ]

    # Phrases that contain a marker but are NOT section headers
    false_positive_patterns = [
        r"conclude[sd]?\s+(?:the\s+|our\s+)?(?:q&a|question)",
        r"end[sed]?\s+(?:the\s+|our\s+)?(?:q&a|question)",
        r"close[sd]?\s+(?:the\s+|our\s+)?(?:q&a|question)",
        r"thank\s+you\s+for\s+(?:the\s+|your\s+)?(?:q&a|question)",
        r"no\s+further\s+questions",
        r"that\s+(?:concludes|ends|completes)\s+",
    ]

    false_positive_re = re.compile(
        "|".join(false_positive_patterns), re.IGNORECASE
    )

    text_lower = text.lower()
    split_idx = len(text)  # default: no Q&A found

    for marker in qa_markers:
        search_start = 0
        while True:
            idx = text_lower.find(marker, search_start) # -1 if none found
            if idx == -1:
                break

            # Grab a window of surrounding text to check for false positives
            window_start = max(0, idx - 60)
            window_end = min(len(text), idx + len(marker) + 60)
            window = text[window_start:window_end]

            if not false_positive_re.search(window):
                # Valid match - check if it's earlier than current best
                if idx < split_idx:
                    split_idx = idx
                break  # no need to keep searching for this marker

            # This occurrence was a false positive — keep searching forward
            search_start = idx + 1

    return {
        "prepared": text[:split_idx],
        "qa": text[split_idx:],
    }

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