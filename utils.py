from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16,
    device_map="auto"
)

def get_vision_response(image_path, user_prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=user_prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result
