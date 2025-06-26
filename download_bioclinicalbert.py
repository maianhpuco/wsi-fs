from transformers import AutoTokenizer, AutoModel
import os

save_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/bioclinicalbert"
os.makedirs(save_path, exist_ok=True)

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token is None:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN not found in environment.")

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_auth_token=token)
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_auth_token=token)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Saved model and tokenizer to: {save_path}")
