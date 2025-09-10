import sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

sys.path.append('/home/fxu/DeepSeek-VL')
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images


from utils import  convert_pil_to_base64

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-base"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

image = Image.open("/home/fxu/DeepSeek-VL/images/training_pipelines.jpg").convert("RGB")
base_img = convert_pil_to_base64(image)
caption = "Make the sky clearer and the grass greener."
conversation = [
        {
            "role": "Assistant", 
            "content": 
            ""
        },
        {
            "role": "User",
            "images": ["data:image;base64," + base_img],
            "content": f"""
                    <image_placeholder>
                    Here are the modification instructions: {caption}\n\n
                    Focus on the fashion item and its attributes such as type, color, pattern, material, shape, fit, and style details.
                    Ignore people and background from the image.
                    Avoid imaginary things. 
                    Be specific and objective so that I can find targeting images based on your description solely without knowing the reference image or modification instructions.
                    Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                    Now, describe how the final fashion item looks after applying the modifications. 
                    Write in 1 to 3 coherent sentences.
                    """
        }
    ]
# conversation = [
#     {
#         "role": "User",
#         "content": f"<image_placeholder>Describe each stage of this image.",
#         "images": [f"data:image;base64,{base_img}"]
#     },
#     {
#         "role": "Assistant",
#         "content": ""
#     }
# ]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=True,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(answer)
