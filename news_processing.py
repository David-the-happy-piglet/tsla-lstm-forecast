import pandas as pd
from transformers import pipeline
from pathlib import Path



INPUT_CSV = "TSLA_news_multi_keyword_2020_2025.csv"  
OUTPUT_CSV =  "news_output_with_score.csv"                 
INPUT_COLUMNS = ["Title"]  # Column names to be used as model input
NEW_COLUMN_NAME = "finbert_score"     # New column name for sentiment score
NEW_COLUMN_NAME_CONFIDENCE = "score_confidence"
NEW_COLUMN_NAME_SCORE = "final_score" 

CHUNK_SIZE = 2048      # Number of rows to process at a time, adjustable based on memory
BATCH_SIZE = 16        # pipeline batch_size, 8~32 is fine for CPU
# ======================================


def build_input_text(row):
    """Combine specified columns into a single text string, skipping NaN."""
    parts = []
    for col in INPUT_COLUMNS:
        val = row.get(col, "")
        if pd.notna(val):
            parts.append(str(val))
    return " ".join(parts)


def label_to_score(label: str) -> int:
    """Map FinBERT output labels to numeric scores."""
    label = label.lower()
    if "positive" in label:
        return 1
    elif "negative" in label:
        return -1
    else:  # neutral or others
        return 0


def main():
    # Initialize FinBERT pipeline, device=-1 for CPU
    nlp = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        framework="pt",
        batch_size=BATCH_SIZE,
    )

    first_chunk = True

    # Read CSV in chunks to avoid loading large file into memory at once
    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE):
        # Construct text list
        texts = chunk.apply(build_input_text, axis=1).tolist()

        # Run FinBERT in batch
        results = nlp(texts, truncation=True)

        # Convert to numeric score
        # scores = [label_to_score(r["label"]) for r in results]
        # scores = [r["score"] for r in results]
        scores = [r["label"] for r in results]
        confidence = [r["score"] for r in results]
        num_score = [label_to_score(r["label"]) for r in results]
        final_score = [num_score[i] * confidence[i] for i in range(len(num_score))]

        # Add columns to chunk
        chunk[NEW_COLUMN_NAME] = scores
        chunk[NEW_COLUMN_NAME_CONFIDENCE] = confidence
        chunk[NEW_COLUMN_NAME_SCORE] = final_score
        
        # Append write to output CSV
        if first_chunk:
            chunk.to_csv(OUTPUT_CSV, index=False, mode="w")
            first_chunk = False
        else:
            chunk.to_csv(OUTPUT_CSV, index=False, mode="a", header=False)


if __name__ == "__main__":
    main()
