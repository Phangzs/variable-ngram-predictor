import numpy as np
import string
import scipy.sparse as sparse
import scipy
import math

# Our json formatter for AllenAI's c4 dataset on HuggingFace
import json_formatter as formatter


datasetsToImport = 1


test = []
for i in range(datasetsToImport):
    # first = "" 
    # second = "" + i % 10
    tests = formatter.extract_text_from_json(f"c4-train.000{i//10}{i%10}-of-01024.json", 200000)
    test.extend(tests)

# Preprocessing to normalize text.
def preprocess_texts(texts):
    processed_texts = []
    for text in texts:
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator).lower()
        processed_texts.append(text)
    return processed_texts

# Build ngram models
def build_ngram_counts(texts, max_n):
    counts = [{} for _ in range(max_n+1)]  
    counter = 0
    for text in texts:
        words = text.split()
        for i in range(len(words)):
            for n in range(1, max_n+1):
                if i + n <= len(words):
                    context = tuple(words[i:i+n-1]) if n > 1 else ()
                    word = words[i+n-1]
                    if context not in counts[n]:
                        counts[n][context] = {}
                    if word not in counts[n][context]:
                        counts[n][context][word] = 0
                    counts[n][context][word] += 1
        print(counter)
        counter += 1
    return counts

# Normalize counts to get probabilies
def convert_counts_to_probs(counts, max_n):
    probs = [{} for _ in range(max_n+1)]
    for n in range(1, max_n+1):
        for context in counts[n]:
            total = sum(counts[n][context].values())
            probs[n][context] = {}
            for word in counts[n][context]:
                probs[n][context][word] = counts[n][context][word] / total
    return probs

# Entropy indicates unpredictability
def compute_entropy(probs_for_context):
    entropy = 0.0
    for p in probs_for_context.values():
        entropy -= p * math.log2(p)
    return entropy

# Decides best ngram order based
def select_best_ngram_order(context_words, probs, counts, max_n):
    best_order = None
    best_score = None
    best_probs = None
    for n in range(max_n, 0, -1):
        context_length = min(n-1, len(context_words))
        context = tuple(context_words[-context_length:]) if context_length > 0 else ()
        if context in probs[n]:
            probs_for_context = probs[n][context]
            entropy = compute_entropy(probs_for_context)
            total_count = sum(counts[n][context].values())
            max_prob = max(probs_for_context.values())
            # Optimization function
            # Favors more predictable contexts with strong probability for next word
            score = (1 / (entropy + 1e-6)) * math.log(total_count + 1) * max_prob
            if best_score is None or score > best_score:
                best_score = score
                best_order = n
                best_probs = probs_for_context
    return best_order, best_probs

# Normalizes probabilities again
# Uses alpha to decrease probability of less likely words more
def adjust_probabilities(probs_dict, alpha):
    # Apply temperature scaling or power transformation to adjust probabilities
    adjusted_probs = {}
    for word, prob in probs_dict.items():
        adjusted_probs[word] = prob ** alpha
    # Normalize the adjusted probabilities
    total = sum(adjusted_probs.values())
    for word in adjusted_probs:
        adjusted_probs[word] /= total
    return adjusted_probs

# Based on best ngram, find the next word
# Also keeps context updated
def generate_text(context_words, probs, counts, max_n, num_words, alpha = 1.0):
    generated = []
    for _ in range(num_words):
        best_order, next_word_probs = select_best_ngram_order(context_words, probs, counts, max_n)
        if not next_word_probs:
            # No available next words
            break
        # Random sampling of words
        adjusted_probs = adjust_probabilities(next_word_probs, alpha)
        words = list(adjusted_probs.keys())
        probabilities = list(next_word_probs.values())
        next_word = np.random.choice(words, p=probabilities)
        generated.append(next_word)
        # Update context for next prediction
        context_words.append(next_word)
        # Keep context length up to max_n - 1
        context_words = context_words[-(max_n-1):]
    return generated

# Preprocess texts
test = preprocess_texts(test)

max_n = 5  # Maximum n-gram order

# Build n-gram counts
counts = build_ngram_counts(test, max_n)

# Convert counts to probabilities
probs = convert_counts_to_probs(counts, max_n)

num_words_to_generate = 20  # Number of words to predict

toPredict = ""

while toPredict != "exit":
    toPredict = input("Input your selected predictee (\"exit\" to exit):\n")
    context_words = toPredict.lower().split()

    num_words_to_generate = 20  # Number of words to predict

    alpha = 2.5

    generated_words = generate_text(context_words, probs, counts, max_n, num_words_to_generate, alpha=alpha)

    print(f"Given the context '{toPredict}', the next {num_words_to_generate} words predicted are:")
    print(' '.join(generated_words))
