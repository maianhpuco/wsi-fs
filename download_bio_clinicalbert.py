from transformers import AutoTokenizer, AutoModel
import os

save_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/bioclinicalbert"
os.makedirs(save_path, exist_ok=True)

# Download and save model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Saved Bio_ClinicalBERT to {save_path}")
