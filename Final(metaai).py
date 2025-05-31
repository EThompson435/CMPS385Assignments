import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Meta AI NLLB model (English → Spanish)...")
meta_model_name = "facebook/nllb-200-distilled-600M"
meta_tokenizer = AutoTokenizer.from_pretrained(meta_model_name)
meta_model = AutoModelForSeq2SeqLM.from_pretrained(meta_model_name).to(device)

print("🧠 Meta AI Translator (English → Spanish)")
print("Type an English sentence to translate. Type 'quit' to exit.\n")

def translate_meta(text):
    # Clean input a bit (strip and remove trailing question marks if desired)
    text = text.strip()
    
    # Encode and generate translation with forced target language token
    inputs = meta_tokenizer(text, return_tensors="pt").to(device)
    translated_tokens = meta_model.generate(
        **inputs,
        forced_bos_token_id=meta_tokenizer.lang_code_to_id["spa_Latn"],
        max_new_tokens=50,
        early_stopping=True,          # Stop generation once EOS token is predicted
        no_repeat_ngram_size=3         # Prevent repeating phrases
    )
    translated_text = meta_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    # Optional: clean trailing unwanted punctuation or fragments if needed
    cleaned = translated_text.strip().rstrip(".-")  
    return cleaned

while True:
    sentence = input("> ").strip()
    if sentence.lower() == 'quit':
        print("Goodbye!")
        break
    if not sentence:
        continue
    output = translate_meta(sentence)
    print("→", output)
