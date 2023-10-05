import re
import nltk
import random
nltk.data.path.append('.')

with open("data/en_US.twitter.txt", "r", encoding="utf-8") as f:
    data = f.read()

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

tokenized_data = get_tokenized_data(data)
kalimat_data = join_tokens_into_sentences(tokenized_data)

# Fungsi untuk membuat n-grams dari data token
def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams

# Fungsi untuk memprediksi kata selanjutnya berdasarkan model n-gram dengan probabilitas
def predict_next_word_with_prob(prefix, model):
    if prefix in model:
        next_words = model[prefix]
        
        # Menghitung probabilitas masing-masing kata
        word_probabilities = []
        total_count = len(next_words)
        for word in next_words:
            count = next_words.count(word)
            probability = count / total_count
            word_probabilities.append((word, probability))
        
        # Memilih kata berdasarkan probabilitas
        selected_word = max(word_probabilities, key=lambda x: x[1])[0]
        return selected_word
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
        next_word = predict_next_word_with_prob(current_prefix, model)
        if next_word is None:
            break
        suggestions.append(next_word)
        current_prefix = current_prefix[1:] + (next_word,)
    
    # Gabungkan kata-kata dalam daftar suggestions menjadi satu kalimat
    completed_sentence = ' '.join(input_text.split() + suggestions)
    
    return completed_sentence

# Contoh penggunaan
input_text = input("Masukkan kata awal (pisahkan dengan spasi): ")
n = 3
num_suggestions = int(input("Masukkan jumlah kata prediksi selanjutnya: "))

completed_sentence = autocomplete(input_text, n, kalimat_data, num_suggestions)
print("Autocomplete suggestions:")
print(completed_sentence)