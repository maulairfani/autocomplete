import re
import nltk
nltk.data.path.append('.')

def split_to_sentences(data):
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]  # Hanya menghapus baris kosong
    return sentences

def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        # Menghapus karakter khusus dan tanda baca, serta mengganti dengan spasi
        sentence = re.sub(r'[^a-zA-Z0-9]', ' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    return tokenized_sentences

def get_tokenized_data(data):
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

# Menggabungkan token-token dalam daftar menjadi kalimat
def join_tokens_into_sentences(tokenized_data):
    sentences = [' '.join(tokens) for tokens in tokenized_data]
    return '\n'.join(sentences)  # Menggabungkan kalimat dengan baris baru (\n)



import random

# Fungsi untuk membuat n-grams dari data token
def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams

# Fungsi untuk memprediksi kata selanjutnya berdasarkan model n-gram
def predict_next_word(prefix, model):
    if prefix in model:
        next_words = model[prefix]
        return random.choice(next_words)
    else:
        return None

# Fungsi untuk membangun model n-gram dari corpus teks
def build_ngram_model(corpus, n):
    tokens = corpus.split()
    ngrams = generate_ngrams(tokens, n)
    
    # Membangun model n-gram
    model = {}
    for ngram in ngrams:
        prefix = tuple(ngram[:-1])
        next_word = ngram[-1]
        if prefix in model:
            model[prefix].append(next_word)
        else:
            model[prefix] = [next_word]
    
    return model

# Fungsi untuk melakukan autocomplete
def autocomplete(input_text, n, corpus, num_suggestions=5):
    model = build_ngram_model(corpus, n)
    current_prefix = tuple(input_text.split()[-n + 1:])
    
    suggestions = []
    for _ in range(num_suggestions):
        next_word = predict_next_word(current_prefix, model)
        if next_word is None:
            break
        suggestions.append(next_word)
        current_prefix = current_prefix[1:] + (next_word,)
    
    # Gabungkan kata-kata dalam daftar suggestions menjadi satu kalimat
    completed_sentence = ' '.join(input_text.split() + suggestions)
    
    return completed_sentence

