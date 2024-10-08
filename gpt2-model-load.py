from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Lade das Modell und den Tokenizer
try:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
except Exception as e:
    print(f"Fehler beim Laden des Modells oder Tokenizers: {e}")
    exit()

# Setze den Pad-Token
tokenizer.pad_token = tokenizer.eos_token

# Definiere den Prompt
prompt = "GPT2 is a model developed by OpenAI."

# Tokenisiere den Prompt und erzeuge Tensoren
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

# Überprüfe die Verwendung von CUDA (GPU), falls vorhanden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
inputs['attention_mask'] = inputs['attention_mask'].to(device)

# Verwende torch.no_grad() um den Gradienten-Speicher zu deaktivieren
with torch.no_grad():
    try:
        # Generiere Text
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
            attention_mask=inputs['attention_mask']
        )
    except Exception as e:
        print(f"Fehler beim Generieren von Tokens: {e}")
        exit()

# Dekodiere die generierten Tokens
gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
print(gen_text)
