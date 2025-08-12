import torch
import base64
import io

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, Blip2ForImageTextRetrieval
from transformers import AutoProcessor as transformer_processor
import torch.nn.functional as F
from torchmetrics.retrieval import RetrievalRecall

from configuration import get_default_config
from fashioniq import get_fashioniq_loader, transform_image
from circo import get_circo_loader
from cirr import get_cirr_loader
from feature_extraction import get_metrics

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
                "Given two images (first: reference, second: target), "
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

if __name__ == "__main__":

    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
    model_id = cfg['TEXT-GENERATION']['MODEL_NAME']

    extractor = cfg['GENERAL']['EXTRACTOR']
    dataset_name = cfg['GENERAL']['DATASET']
    top_k = cfg['GENERAL']['TOP_K']
    print(f"Using {extractor} for feature extraction using {dataset_name}")

    extractor_id = cfg[extractor]['MODEL_NAME']
    img_transform=transform_image(cfg[extractor]['IMAGE_SIZE'],
                                cfg[extractor]['IMAGE_MEAN'],
                                cfg[extractor]['IMAGE_STD'])
    if extractor.lower() == 'blip2':
        feature_extraction_model = Blip2ForImageTextRetrieval.from_pretrained(extractor_id, torch_dtype=torch.float16).to(device)
    else:
        feature_extraction_model = AutoModel.from_pretrained(extractor_id).to(device)
    feature_processor = transformer_processor.from_pretrained(extractor_id)

    text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map={"": device}
        )
    processor = AutoProcessor.from_pretrained(model_id, padding_side='left', use_fast=True)


    if dataset_name.lower() == "fashioniq":
        dataloader = get_fashioniq_loader(
            output_dir=cfg['FashionIQ']['OUTPUT_DIR'],
            batch_size=cfg['GENERAL']['BATCH_SIZE'],
            transform=img_transform
        )
    elif dataset_name.lower() == "circo":
        dataloader = get_circo_loader(
            batch_size=cfg['GENERAL']['BATCH_SIZE'],
            split='val',
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=img_transform
        )
    elif dataset_name.lower() == "cirr":
        dataloader = get_cirr_loader(
            batch_size=cfg['GENERAL']['BATCH_SIZE'],
            split='val',
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=img_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    caption_feat = []
    modification_feat = []
    description_feat = []
    tar_tensor_feat = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            target = batch['target_pil']
            reference = batch['reference_pil']
            caption = batch['caption']
            target_tensor = batch['target']

            messages = list(map(lambda x: generate_messages(*x),zip(target, reference, caption)))
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
                all_text_inputs = feature_processor(text=caption+generated_text, 
                                                    return_tensors="pt", 
                                                    padding=True, 
                                                    max_length=64,
                                                    truncation=True
                                                ).to(device)
                gen_text_feat = feature_extraction_model.get_text_features(**all_text_inputs)
            elif extractor.lower() == 'clip':
                all_text_inputs = feature_processor(caption+generated_text, 
                                                    return_tensors="pt", 
                                                    padding=True, 
                                                    truncation=True
                                                ).to(device)
                gen_text_feat = feature_extraction_model.get_text_features(**all_text_inputs)
            elif extractor.lower() == 'blip2':
                all_text_inputs = feature_processor(text=caption+generated_text, 
                                                    images=target,
                                                    return_tensors="pt",
                                                    padding=True
                                                ).to(device, torch.float16)
                gen_text_feat = feature_extraction_model(**all_text_inputs, use_image_text_matching_head=False)

            if extractor.lower() == 'blip2':
                steps = gen_text_feat.text_embeds.size(0)//3
                caption_feat.append(F.normalize(gen_text_feat.text_embeds[:steps, :], p=2, dim=-1))
                modification_feat.append(F.normalize(gen_text_feat.text_embeds[steps:steps*2, :], p=2, dim=-1))
                description_feat.append(F.normalize(gen_text_feat.text_embeds[steps*2:, :], p=2, dim=-1))
                tar_tensor_feat.append(F.normalize(gen_text_feat.image_embeds, p=2, dim=-1))
            else:
                steps = gen_text_feat.size(0)//3
                caption_feat.append(gen_text_feat[:steps, :])
                modification_feat.append(gen_text_feat[steps:steps*2, :])
                description_feat.append(gen_text_feat[steps*2:, :])
                tar_tensor_feat.append(feature_extraction_model.get_image_features(pixel_values=target_tensor.to(device)))

    caption_feat = torch.cat(caption_feat, dim=0)
    modification_feat = torch.cat(modification_feat, dim=0)
    description_feat = torch.cat(description_feat, dim=0)
    tar_tensor_feat = torch.cat(tar_tensor_feat, dim=0)

    #compute recall for generated modification ---> caption
    if dataset_name.lower() == "circo":
        metric = 'map'
    else:
        metric = 'recall'
    for k in top_k:
        metric_val = get_metrics(modification_feat, caption_feat, k=k, metrics=metric)
        print(f'{metric.capitalize()}@{k}: {metric_val:.2f} when using generated modification ---> real modification')

    #compute recall for generated description ---> target image
    if extractor.lower() == 'blip2':
        cos, _ = torch.matmul(description_feat.unsqueeze(0), tar_tensor_feat.transpose(1,2)).max(dim=-1)
        cos = cos.T
        for k in top_k:
            compute_recall=RetrievalRecall(top_k=k)
            targets=torch.diag(torch.ones(cos.size(0), dtype=torch.long)).to(cos.device)
            indexes = torch.arange(cos.size(0), dtype=torch.long).unsqueeze(1).expand(*cos.size()).to(cos.device)
            recall=compute_recall(cos.flatten(),targets.flatten(),indexes=indexes.flatten())
            print(f'Recall@{k}: {recall:.2f} when using generated description ---> target images')

        ##### still need to implement mAP for circo dataset!!!!
    else:
        for k in top_k:
            recall = get_metrics(description_feat, tar_tensor_feat, k=k, metrics=metric)
            print(f'Recall@{k}: {recall:.2f} when using generated description ---> target images')