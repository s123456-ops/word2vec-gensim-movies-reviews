
from pathlib import Path
import re
import time

import pandas as pd
import gensim
from gensim.models import Word2Vec


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent


def resolve_case_insensitive_dir(name):
    for candidate in [
        PROJECT_DIR / name,
        PROJECT_DIR / name.lower(),
        PROJECT_DIR / name.upper(),
    ]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return PROJECT_DIR / name


DATA_DIR = resolve_case_insensitive_dir("Data")
RESULTS_DIR = resolve_case_insensitive_dir("Results")
RESULTS_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "movies_metadata.csv"
MODEL_PATH = RESULTS_DIR / "movie_word2vec_clean.model"
RESULTS_PATH = RESULTS_DIR / "movie_results.txt"


# --------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------

def tokenize(text):
    text = text.lower()

    # Normalize common possessives: "Andy's" -> "andy"
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)

    # Keep contractions together where possible
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    # Remove one-letter tokens except "i"
    words = [
        word for word in words
        if len(word) > 1 or word == "i"
    ]

    return words


# --------------------------------------------------
# 3. Load and preprocess the dataset
# --------------------------------------------------

print("Gensim version:", gensim.__version__)
print("\nLoading movie dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

movies = pd.read_csv(DATA_PATH, low_memory=False)

print("Original number of rows:", len(movies))

movies = movies.dropna(subset=["overview"]).copy()
movies["overview"] = movies["overview"].astype(str)
movies = movies[
    movies["overview"].str.strip() != ""
].copy()

print("Rows with valid summaries:", len(movies))

# Create tokens BEFORE filtering on the tokens column
movies["tokens"] = movies["overview"].apply(tokenize)

movies = movies[
    movies["tokens"].map(len) > 0
].copy()

sentences = movies["tokens"].tolist()

print("Rows after preprocessing:", len(sentences))

total_tokens = sum(len(sentence) for sentence in sentences)
print("Total tokens:", total_tokens)

if not sentences:
    raise ValueError("No usable movie summaries were found.")

print("\nExample original summary:")
print(movies["overview"].iloc[0][:300])

print("\nExample tokenized summary:")
print(sentences[0][:30])

print("\nTokenizer test:")
print(tokenize("Andy's phone is good, but don't buy a low-quality screen!"))


# --------------------------------------------------
# 4. Train Word2Vec (Skip-gram)
# --------------------------------------------------

print("\nStarting Word2Vec Skip-gram training...")

start_time = time.time()

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=10,
    seed=42
)

training_time = time.time() - start_time

print("\nTraining completed!")
print(f"Training time: {training_time:.2f} seconds")
print("Vocabulary size:", len(model.wv))
print("Vector dimensions:", model.wv.vector_size)

model.save(str(MODEL_PATH))
print("Model saved to:", MODEL_PATH)


# --------------------------------------------------
# 5. Evaluate the model
# --------------------------------------------------

print("\n--- Word2Vec Evaluation: Movies ---")

output_lines = [
    "WORD2VEC RESULTS - MOVIE SUMMARIES",
    f"Gensim version: {gensim.__version__}",
    f"Original rows: {len(pd.read_csv(DATA_PATH, low_memory=False))}",
    f"Rows after preprocessing: {len(sentences)}",
    f"Total tokens: {total_tokens}",
    f"Vocabulary size: {len(model.wv)}",
    f"Vector dimensions: {model.wv.vector_size}",
    f"Training time: {training_time:.2f} seconds",
    "Architecture: Skip-gram",
    "Window: 5",
    "Minimum word count: 5",
    "Epochs: 10",
    ""
]

test_words = ["love", "war", "family", "death", "man"]

for word in test_words:
    if word in model.wv:
        print(f"\nWords most similar to '{word}':")
        output_lines.append(f"Words most similar to '{word}':")

        for similar_word, score in model.wv.most_similar(
            word, topn=5
        ):
            line = f"  {similar_word}: {score:.4f}"
            print(line)
            output_lines.append(line)
    else:
        message = f"\n'{word}' is not in the vocabulary."
        print(message)
        output_lines.append(message)

word1 = "love"
word2 = "family"

if word1 in model.wv and word2 in model.wv:
    score = model.wv.similarity(word1, word2)
    message = (
        f"\nCosine similarity ({word1}, {word2}): "
        f"{score:.4f}"
    )
    print(message)
    output_lines.append(message)

RESULTS_PATH.write_text(
    "\n".join(output_lines),
    encoding="utf-8"
)

print("\nResults saved to:", RESULTS_PATH)
print("\nMovie experiment completed!")
