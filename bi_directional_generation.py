import torch

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from configuration import get_default_config
from feature_extraction import get_metrics, get_feature_extractor
from dataloaders import get_dataloader
from text_generation import generate_target_description, convert_pil_to_base64

def generate_image_description(target_image):
    # target_description = [
    #     {
    #         "role": "system", 
    #         "content": (
    #             "You are an expert at visual perception. "
    #             "Given an image, you can describe the image in an accurate, detailed and complete natural-language description. "
    #             "Include colors, lighting, textures, positions, objects, people, and atmosphere. "
    #             "Write in clear, logical, full, and complete English sentences."
    #         )
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(target_image)},
    #             {
    #                 "type": "text", 
    #                 "text": (
    #                     "Now, describe what you see from the given image."
    #                 )
    #             }
    #         ],
    #     }
    # ]
    target_description = [
        {
            "role": "system", 
            "content": (
                "You are an expert in detailed visual perception. "
                "Given an image, you must produce a single, continuous, natural-language description. "
                "Always use complete, well-formed English sentences, not fragments or bullet points. "
                "Describe the image thoroughly, including objects, people, colors, lighting, textures, positions, and atmosphere. "
                "Your response must be a coherent multi-sentence paragraph, not a list."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(target_image)},
                {
                    "type": "text", 
                    "text": "Please describe this image in full English sentences."
                }
            ],
        }
    ]

    return target_description

def main(cfg):
    device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
    model_id = cfg['TEXT-GENERATION']['MODEL_NAME']

    extractor = cfg['GENERAL']['EXTRACTOR']
    dataset_name = cfg['GENERAL']['DATASET']
    top_k = cfg['GENERAL']['TOP_K']
    print(f"Using {extractor} for feature extraction using {dataset_name}")

    if extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
        feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg)
    else:
        feature_extraction_model, tokenizer = get_feature_extractor(cfg)
    feature_extraction_model.eval()
    feature_extraction_model.to(device)

    text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, 
                                                                               torch_dtype=torch.bfloat16, 
                                                                               device_map={"": device}, 
                                                                               attn_implementation='flash_attention_2'
                                                                               ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, 
                                              padding_side='left', 
                                              use_fast=True
                                              )

    dataloader = get_dataloader(cfg)

    caption_feat = []
    description_feat = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            target_pil = batch['target_pil']
            reference_pil = batch['reference_pil']
            caption = batch['caption']

            target_description = list(map(lambda x: generate_target_description(*x),zip(reference_pil, caption)))
            image_description = list(map(generate_image_description, target_pil))

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in target_description+image_description
            ]
            image_inputs, video_inputs = process_vision_info(target_description+image_description)
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
            print(output_texts)

            if extractor.lower() == 'siglip2':
                all_text_inputs = tokenizer(text=output_texts, 
                                                    return_tensors="pt", 
                                                    padding=True, 
                                                    max_length=64,
                                                    truncation=True
                                                ).to(device)
                gen_feat = feature_extraction_model.get_text_features(**all_text_inputs)
            elif extractor.lower() == 'clip':
                all_text_inputs = tokenizer(output_texts, 
                                                    return_tensors="pt", 
                                                    padding=True, 
                                                    truncation=True
                                                ).to(device)
                gen_feat = feature_extraction_model.get_text_features(**all_text_inputs)
            elif extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
                all_inputs = tokenizer(output_texts,
                                       context_length=feature_extraction_model.context_length
                                       ).to(device)
                gen_feat = feature_extraction_model.encode_text(all_inputs)
            else:
                raise ValueError(f"Unsupported extractor: {extractor}")

            steps = gen_feat.size(0)//2
            caption_feat.append(gen_feat[:steps, :])
            description_feat.append(gen_feat[steps:, :])

    caption_feat = torch.cat(caption_feat, dim=0)
    description_feat = torch.cat(description_feat, dim=0)

    if dataset_name.lower() == "circo":
        metric = 'map'
    else:
        metric = 'recall'
    for k in top_k:
        #compute recall for generated description ---> target image
        metric_val = get_metrics(caption_feat, description_feat, k=k, target_length=[1]*description_feat.size(0), metrics=metric)
        print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated description ---> target images')


if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    main(cfg)