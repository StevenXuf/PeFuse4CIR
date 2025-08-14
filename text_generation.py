import torch
import base64
import io

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from configuration import get_default_config
from feature_extraction import get_metrics, get_feature_extractor
from dataloaders import get_dataloader

def convert_pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_messages(target_image, reference_image, caption):
    text_modification = [
        {
            "role": "system", 
            "content": (
                "You are an expert at comparing images and identifying visual differences. "
                "Given two images (first: _pil, second: target), "
                "describe all the changes needed to transform the first image into the second. "
                "Be complete and specific—mention differences in objects, colors, lighting, textures, positions, sizes, and background details. "
                "Only describe every visible changes between the images. "
                "Write in clear, full, and complete sentences in English."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": "data:image;base64," + convert_pil_to_base64(reference_image)
                },
                {
                    "type": "image", 
                    "image": "data:image;base64," + convert_pil_to_base64(target_image)
                },
                {
                    "type": "text", 
                    "text": (
                        "Compare the first image to the second image. "
                        "Describe exactly what has been changed, added, or removed in order to make the first image match the second."
                    )
                }
            ],
        }
    ]


    target_description = [
        {
            "role": "system", 
            "content": (
                "You are an expert at visual imagination. "
                "Given an image and modification instructions, you will mentally apply the changes "
                "and then produce a accurate, and complete natural-language description of "
                "what the resulting image looks like. "
                "Only describe the final modified scene. "
                "Include colors, lighting, textures, positions, objects, people, and atmosphere. "
                "Write in clear, full, and complete sentences in English."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": (
                        f"Here are the modification instructions: {caption}\n\n"
                        "Now, describe how the final image looks in natural language."
                    )
                }
            ],
        }
    ]

    return text_modification, target_description

def main(cfg):
    device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
    model_id = cfg['TEXT-GENERATION']['MODEL_NAME']

    extractor = cfg['GENERAL']['EXTRACTOR']
    dataset_name = cfg['GENERAL']['DATASET']
    top_k = cfg['GENERAL']['TOP_K']
    print(f"Using {extractor} for feature extraction using {dataset_name}")

    if extractor.lower() == 'openvision':
        feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg)
    else:
        feature_extraction_model, tokenizer = get_feature_extractor(cfg)

    text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map={"": device}
        )
    processor = AutoProcessor.from_pretrained(model_id, padding_side='left', use_fast=True)

    dataloader = get_dataloader(cfg)

    caption_feat = []
    modification_feat = []
    description_feat = []
    tar_tensor_feat = []
    target_length = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            target_pil = batch['target_pil']
            reference_pil = batch['reference_pil']
            reference_tensor = batch['reference_img']
            caption = batch['caption']
            target_tensor = batch['target_img']
            all_target_pil = batch['all_target_pil']
            all_target_tensor = batch['all_target_img']
            target_length.extend(batch['all_target_length'])

            messages = list(map(lambda x: generate_messages(*x),zip(target_pil, reference_pil, caption)))
            text_modification = [msg[0] for msg in messages]
            target_description = [msg[1] for msg in messages]

            generated_text = []
            for text_info in [text_modification, target_description]:
                texts = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in text_info
                ]
                image_inputs, video_inputs = process_vision_info(text_info)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)

                # Batch Inference
                generated_ids = text_generation_model.generate(**inputs, max_new_tokens=76)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                generated_text.extend(output_texts)

            if extractor.lower() == 'siglip2':
                all_text_inputs = tokenizer(text=caption+generated_text, 
                                                    return_tensors="pt", 
                                                    padding=True, 
                                                    max_length=64,
                                                    truncation=True
                                                ).to(device)
                gen_feat = feature_extraction_model.get_text_features(**all_text_inputs)
            elif extractor.lower() == 'clip':
                all_text_inputs = tokenizer(caption+generated_text, 
                                                    return_tensors="pt", 
                                                    padding=True, 
                                                    truncation=True
                                                ).to(device)
                gen_feat = feature_extraction_model.get_text_features(**all_text_inputs)
            elif extractor.lower() == 'openvision':
                all_inputs = tokenizer(caption+generated_text,
                                       context_length=feature_extraction_model.context_length
                                       ).to(device)
                gen_feat = feature_extraction_model.encode_text(all_inputs)

            steps = gen_feat.size(0)//3
            caption_feat.append(gen_feat[:steps, :])
            modification_feat.append(gen_feat[steps:steps*2, :])
            description_feat.append(gen_feat[steps*2:, :])
            if extractor.lower() == 'openvision':
                img_feat = feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in all_target_pil],dim=0).to(device))
            else:
                img_feat = feature_extraction_model.get_image_features(pixel_values=all_target_tensor.to(device))
            tar_tensor_feat.append(img_feat)

    print(target_length)
    caption_feat = torch.cat(caption_feat, dim=0)
    modification_feat = torch.cat(modification_feat, dim=0)
    description_feat = torch.cat(description_feat, dim=0)
    tar_tensor_feat = torch.cat(tar_tensor_feat, dim=0)

    if dataset_name.lower() == "circo":
        metric = 'map'
    else:
        metric = 'recall'
    for k in top_k:
        #compute recall for generated modification ---> caption
        metric_val = get_metrics(modification_feat, caption_feat, k=k, target_length=target_length, metrics=metric)
        print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated modification ---> real modification')

        #compute recall for generated description ---> target image
        metric_val = get_metrics(description_feat, tar_tensor_feat, k=k, target_length=target_length, metrics=metric)
        print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated description ---> target images')


if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    main(cfg)