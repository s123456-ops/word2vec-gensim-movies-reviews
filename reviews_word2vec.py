
from pathlib import Path
import gzip
import json
import re
import time

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

DATA_PATH = DATA_DIR / "reviews.json.gz"
MODEL_PATH = RESULTS_DIR / "reviews_word2vec_corrected.model"
RESULTS_PATH = RESULTS_DIR / "reviews_results_corrected.txt"


# --------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------

# Map known misspellings to their intended words.
CORRECTIONS = {
    # Variants of "quality"
    "quaility": "quality",
    "qulity": "quality",
    "quailty": "quality",
    "quility": "quality",
    "qualty": "quality",
    "qality": "quality",
    "quallity": "quality",
    "phoen": "phone",
    "pone": "phone",
    "phome": "phone",
    "sreen": "screen",
}

# Split only joined words identified in the model's results.
JOINED_WORDS = {
    "qualitygreat": "quality great",
    "qualityi": "quality i",
    "qualitythe": "quality the",
    "phoneit": "phone it",
}


def tokenize(text):
    text = text.lower()

    # Normalize possessives, e.g. "Andy's" -> "andy"
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)

    # Split known joined words before tokenization
    for joined, separated in JOINED_WORDS.items():
        text = re.sub(
            rf"\b{re.escape(joined)}\b",
            separated,
            text
        )

    # Keep contractions together where possible
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    # Normalize known spelling variants
    words = [CORRECTIONS.get(word, word) for word in words]

    # Remove one-letter tokens except the pronoun "i"
    words = [
        word for word in words
        if len(word) > 1 or word == "i"
    ]

    return words


# --------------------------------------------------
# 3. Test preprocessing
# --------------------------------------------------

print("Gensim version:", gensim.__version__)

test_text = (
    "The quaility is good, the qulity is great, "
    "and the quailty is excellent. qualitygreat! "
    "qualityi like it. qualitythe phone. "
    "This phoen is good and phoneit works."
)

print("\nTokenizer test:")
print(tokenize(test_text))


# --------------------------------------------------
# 4. Load and preprocess reviews
# --------------------------------------------------

print("\nReading and preprocessing reviews...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

sentences = []
reviews_read = 0
empty_reviews = 0
example_original = ""

with gzip.open(DATA_PATH, "rt", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        review = json.loads(line)
        text = review.get("reviewText", "")

        if not isinstance(text, str) or not text.strip():
            empty_reviews += 1
            continue

        if not example_original:
            example_original = text

        tokens = tokenize(text)

        if tokens:
            sentences.append(tokens)
            reviews_read += 1
        else:
            empty_reviews += 1

print("\nPreprocessing completed!")
print("Reviews retained:", reviews_read)
print("Empty or unusable reviews:", empty_reviews)

total_tokens = sum(len(sentence) for sentence in sentences)
print("Total tokens:", total_tokens)

if not sentences:
    raise ValueError("No usable reviews were found.")

print("\nExample original review:")
print(example_original[:300])

print("\nExample tokenized review:")
print(sentences[0][:30])


# --------------------------------------------------
# 5. Train Word2Vec using Skip-gram
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
# 6. Evaluate the model
# --------------------------------------------------

print("\n--- Word2Vec Evaluation: Corrected Reviews ---")

output_lines = [
    "WORD2VEC RESULTS - CORRECTED AMAZON REVIEWS",
    f"Gensim version: {gensim.__version__}",
    f"Reviews retained: {reviews_read}",
    f"Empty or unusable reviews: {empty_reviews}",
    f"Total tokens: {total_tokens}",
    f"Vocabulary size: {len(model.wv)}",
    f"Vector dimensions: {model.wv.vector_size}",
    f"Training time: {training_time:.2f} seconds",
    "Architecture: Skip-gram",
    "Window: 5",
    "Minimum word count: 5",
    "Epochs: 10",
    "Known spelling variants normalized: yes",
    "Known joined words split: yes",
    ""
]

test_words = ["phone", "battery", "screen", "quality", "good"]

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

word1 = "battery"
word2 = "charging"

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
print("\nReview experiment completed!")
