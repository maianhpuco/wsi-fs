import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os

def run_quilt_llava(image_path, prompt="### Explain this pathology slide"):
    model_id = "wisdomik/Quilt-Llava-v1.5-7b"

    # Load model + processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.float16)

    # Generate
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return response

# Example usage
if __name__ == "__main__":
    img_path = "path/to/your/image.png"
    print(run_quilt_llava(img_path))
