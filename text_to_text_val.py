import torch
import fire

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig, set_seed
from qwen_vl_utils import process_vision_info

from configuration import get_default_config
from feature_extraction import get_metrics, get_feature_extractor
from dataloaders import get_dataloader
from text_generation_val import fashioniq_eval
from prompts import get_composed_prompts, get_target_prompts

def main(cfg, **kwargs):
    device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
    model_id = cfg['TEXT-GENERATION']['MODEL_NAME']

    if kwargs.get('TEMPERATURE'):
        temperature = kwargs['TEMPERATURE']
    else:
        temperature = cfg['TEXT-GENERATION']['GLOBAL']['TEMPERATURE']
    if kwargs.get('TOP_P'):
        top_p = kwargs['TOP_P']
    else:
        top_p = cfg['TEXT-GENERATION']['GLOBAL']['TOP_P']
    if kwargs.get('TOP_K'):
        llm_top_k = kwargs['TOP_K']
    else:
        llm_top_k = cfg['TEXT-GENERATION']['GLOBAL']['TOP_K']
    if kwargs.get('MAX_NEW_TOKENS'):
        max_new_tokens = kwargs['MAX_NEW_TOKENS']
    else:
        max_new_tokens = cfg['TEXT-GENERATION']['GLOBAL']['MAX_NEW_TOKENS']

    if kwargs.get('BATCH_SIZE'):
        batch_size = kwargs['BATCH_SIZE']
    else:
        batch_size = cfg['GENERAL']['BATCH_SIZE']
    if kwargs.get('EXTRACTOR'):
        extractor = kwargs['EXTRACTOR']
    else:
        extractor = cfg['GENERAL']['EXTRACTOR']
    if kwargs.get('DATASET'):
        dataset_name = kwargs['DATASET']
    else:
        dataset_name = cfg['GENERAL']['DATASET']
    if kwargs.get('SPLIT'):
        split = kwargs['SPLIT']
    else:
        split = cfg['GENERAL']['SPLIT']
    top_k = cfg['GENERAL']['TOP_K']
    print(f"Using {extractor} for feature extraction using {dataset_name}")

    if extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
        feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, extractor=extractor)
    else:
        feature_extraction_model, tokenizer = get_feature_extractor(cfg, extractor=extractor)
    feature_extraction_model.eval()
    feature_extraction_model.to(device)
    gen_config = GenerationConfig(do_sample=True,
                                  temperature=temperature,
                                  top_p=top_p,
                                  top_k=llm_top_k,
                                  max_new_tokens=max_new_tokens
                                  )
    text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, 
                                                                               torch_dtype=torch.bfloat16, 
                                                                               device_map={"": device}, 
                                                                               attn_implementation='flash_attention_2'
                                                                               ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, 
                                              padding_side='left', 
                                              use_fast=True
                                              )

    dataloader = get_dataloader(cfg, batch_size=batch_size, dataset_name=dataset_name, extractor_name=extractor)

    caption_feat = []
    description_feat = []
    target_length = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            target_pil = batch['target_pil']
            all_target_pil = batch['all_target_pil']
            reference_pil = batch['reference_pil']
            caption = batch['caption']
            target_length.extend(batch['all_target_length'])

            composed_description = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))
            target_description = list(map(lambda x: get_target_prompts(dataset_name, *x), zip(all_target_pil)))

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in composed_description+target_description
            ]
            image_inputs, video_inputs = process_vision_info(composed_description+target_description)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                padding_side="left"
            )
            inputs = inputs.to(device)

            # Batch Inference
            generated_ids = text_generation_model.generate(**inputs,
                                                           generation_config=gen_config
                                                           )
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

            steps = len(composed_description)
            caption_feat.append(gen_feat[:steps, :])
            description_feat.append(gen_feat[steps:, :])

    print(target_length)
    caption_feat = torch.cat(caption_feat, dim=0)
    description_feat = torch.cat(description_feat, dim=0)

    if dataset_name.lower() == "circo":
        metric = 'map'
    else:
        metric = 'recall'
    for k in top_k:
        #compute recall for generated description ---> target image
        if dataset_name.lower() == "fashioniq" and split == 'val':
            fashioniq_eval(dataloader, caption_feat, description_feat, target_length, k)
        else:
            metric_val = get_metrics(caption_feat, description_feat, k=k, target_length=target_length, metrics=metric)
            print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated description ---> target description')

def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    set_seed(cfg['GENERAL']['SEED'])
    main(cfg, **kwargs)

if __name__ == "__main__":
    fire.Fire(launch)