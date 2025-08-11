import torch
from PIL import Image
import requests
from transformers import AutoProcessor, Blip2ForImageTextRetrieval, Blip2Model
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(images=image, text=texts, return_tensors="pt").to(device, torch.float16)
itc_out = model(**inputs, use_image_text_matching_head=False)
print(itc_out.text_embeds.size())
vals = torch.mean(itc_out.image_embeds,dim=1)
print(vals.size())
logits_per_image = itc_out.logits_per_image  # this is the image-text similarity score
print(logits_per_image)
cosine = pairwise_cosine_similarity(itc_out.text_embeds, vals)
print(cosine)