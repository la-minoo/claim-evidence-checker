from transformers import AutoTokenizer 
from datasets import load_dataset 

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base") 
print("Tokenizer OK:", tokenizer.model_max_length) 

ds1 = load_dataset("allenai/scifact", "claims") 
print("SciFact splits:", ds1) 

ds2 = load_dataset("multi_nli", split="train[:100]") 
print("MultiNLI sample:", ds2[0]["premise"][:60])