import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLInstructPix2PixPipeline
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration

from utils import get_gpu_memory, get_default_config

cfg = get_default_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
before = get_gpu_memory(device)

model_id = cfg['TEXT-GENERATION']['LLAVA']['MODEL_NAME']
text_generation_model = LlavaForConditionalGeneration.from_pretrained(model_id, 
                                                                torch_dtype=torch.bfloat16, 
                                                                device_map={"": device}, 
                                                                # attn_implementation='flash_attention_2'
                                                                ).to(device)
after = get_gpu_memory(device)
print(f"momery taken: {(after - before) / 1024**2:.2f} MB")