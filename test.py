import torch
from PIL import Image
import requests
from transformers import AutoProcessor, Blip2ForImageTextRetrieval, Blip2Model
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(images=[image,image,image], text=texts, return_tensors="pt").to(device, torch.float16)
itc_out = model(**inputs, use_image_text_matching_head=False)
print(itc_out.text_embeds.size())
print(itc_out.image_embeds.size())
text_feat = F.normalize(itc_out.text_embeds, p=2, dim=-1)
image_feat = F.normalize(itc_out.image_embeds, p=2, dim=-1)

cos,_ =torch.matmul(text_feat.unsqueeze(0), image_feat.transpose(1,2)).max(dim=-1)
print(cos.size())
print(cos)