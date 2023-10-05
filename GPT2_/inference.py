from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to load model
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

# Function to load tokenizer
def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# Function to generate text
def generate_text(sequence, max_length, model_path="autocomplete_gpt2"):

    model = load_model(model_path)
    tokenizer = load_tokenizer("gpt2")
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    sequence = input("Masukkan kalimat: ")
    max_len = int(input("Masukkan jumlah kata yang ingin digenerate: "))
    model_path = "autocomplete_gpt2"
    generate_text(model_path, sequence, max_len)