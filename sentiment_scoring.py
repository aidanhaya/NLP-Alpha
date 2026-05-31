from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class FinBERTScorer:
    def __init__(self):
        model_name = "ProsusAI/finbert" # best publicly available FinBERT
        # downloads tokenizer and model weights from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval() # puts model in evaluation (instead of training) mode
        self.labels = ["positive", "negative", "neutral"] # defines sentiment classes

    def score_sentence(self, text: str) -> dict:
        inputs = self.tokenizer(
            # tokenizes sentence into a pytorch tensor
            text, return_tensors="pt",
            # truncates text longer than 512 tokens (BERT's limit)
            # pads shorter sentences with 0-tokens so batches are same length
            truncation=True, max_length=512, padding=True
        )
        with torch.no_grad(): # doesn't track gradients
            # projects 768-dim BERT hidden state vectors onto a 3-dim vector
            # that 3-dim vector represents positive, negative, neutral sentiment
            # those values are logits - unnormalized probabilities
            logits = self.model(**inputs).logits
        # runs softmax on logits to get normalized probabilities
        # squeeze() removes batch dimension, converts 2-dim matrix to a numpy vector
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        return dict(zip(self.labels, probs)) # zips label names with their probabilities

    def score_document(self, sentences: list) -> dict:
        """
        Score a list of sentences and aggregate.
        Returns mean sentiment score and component probabilities.
        """
        if not sentences:
            return {"score": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0, "n_sentences": 0}

        scores = [self.score_sentence(s) for s in sentences]

        mean_pos = np.mean([s["positive"] for s in scores])
        mean_neg = np.mean([s["negative"] for s in scores])
        mean_neu = np.mean([s["neutral"] for s in scores])

        # Net sentiment: positive probability minus negative probability
        net_score = float(mean_pos - mean_neg)

        return {
            "score": net_score,  # ranges from -1 to +1
            "positive": float(mean_pos),
            "negative": float(mean_neg),
            "neutral": float(mean_neu),
            "n_sentences": len(sentences),
        }


def score_transcript(tokenized_text: dict, ticker: str, date: str, scorer: FinBERTScorer) -> dict:
    prepared_sentences = tokenized_text["prepared"]
    qa_sentences = tokenized_text["qa"]

    prepared_score = scorer.score_document(prepared_sentences)
    qa_score = scorer.score_document(qa_sentences)

    # Q&A weighted more heavily - it's less scripted
    # checks that a qa section was found - if not, all weight given to prepared_score
    if qa_score["n_sentences"] > 0:
        composite = 0.4 * prepared_score["score"] + 0.6 * qa_score["score"]
    else:
        composite = prepared_score["score"]

    return {
        "ticker": ticker,
        "date": date,
        "prepared": prepared_score,
        "qa": qa_score,
        "composite": composite,
    }