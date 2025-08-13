import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from refinedfashioniq import transform_image

img_transform = transform_image(
    image_size=224,
    IMAGENET_MEAN=[0.485, 0.456, 0.406],
    IMAGENET_STD=[0.229, 0.224, 0.225]
)
model, preprocess = create_model_from_pretrained('hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224')
tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224')

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
image = preprocess(image).unsqueeze(0)
print(image.size())

text = tokenizer(["a diagram", "a dog", "a cat", "a beignet"], context_length=model.context_length)

with torch.no_grad(), torch.amp.autocast('cuda:1'):
    image_features = model.encode_image(torch.randn(3, 3, 224, 224))
    text_features = model.encode_text(text)
    print(text_features.size())
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    sim = image_features @ text_features.T

print(sim)  # prints: [[0., 0., 0., 1.0]]
