import torch
import fire
import numpy as np
import json
import gc

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig, set_seed
from qwen_vl_utils import process_vision_info
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from configuration import get_default_config
from feature_extraction import get_feature_extractor, get_metrics
from dataloaders import get_dataloader
from prompts import get_composed_prompts, get_target_prompts

def fashioniq_eval(dataloader, generated_target_features, target_features, target_length, k):
    n_shirt, n_dress, n_toptee = dataloader.dataset.length
    shirt_metric_val = get_metrics(generated_target_features[:n_shirt, :], target_features[:n_shirt, :], k=k, target_length=target_length[:n_shirt], metrics='recall')
    dress_metric_val = get_metrics(generated_target_features[n_shirt:n_shirt+n_dress, :], target_features[n_shirt:n_shirt+n_dress, :], k=k, target_length=target_length[n_shirt:n_shirt+n_dress], metrics='recall')
    toptee_metric_val = get_metrics(generated_target_features[n_shirt+n_dress:, :], target_features[n_shirt+n_dress:, :], k=k, target_length=target_length[n_shirt+n_dress:], metrics='recall')
    print(f'Recall@{k}: {shirt_metric_val:.2f}% for shirt')
    print(f'Recall@{k}: {dress_metric_val:.2f}% for dress')
    print(f'Recall@{k}: {toptee_metric_val:.2f}% for toptee')
    print(f'Recall@{k}: {(shirt_metric_val+dress_metric_val+toptee_metric_val)/3:.2f}% for all categories')

def generate_texts(messages, gen_config, processor, text_generation_model):
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        padding_side='left'
    )
    inputs = inputs.to(text_generation_model.device)

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
    return output_texts

def extract_text_features(input_texts, extractor, tokenizer, feature_extraction_model):
    if extractor.lower() == 'siglip2' or extractor.lower() == 'clip':
        all_text_inputs = tokenizer(text=input_texts, 
                                            return_tensors="pt", 
                                            padding=True, 
                                            max_length=feature_extraction_model.config.text_config.max_position_embeddings,
                                            truncation=True
                                        ).to(feature_extraction_model.device)
        gen_feat = feature_extraction_model.get_text_features(**all_text_inputs)
    elif extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
        all_inputs = tokenizer(input_texts,
                            context_length=feature_extraction_model.context_length
                            ).to(next(feature_extraction_model.parameters()).device)
        gen_feat = feature_extraction_model.encode_text(all_inputs)
    else:
        raise ValueError(f"Unsupported extractor: {extractor}")
    return gen_feat

def extract_image_features(pil, extractor, feature_extraction_model, img_preprocess):
    if extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
        img_inputs = torch.cat([img_preprocess(img).unsqueeze(0) for img in pil],dim=0).to(next(feature_extraction_model.parameters()).device)
        img_feat = feature_extraction_model.encode_image(img_inputs)
    elif extractor.lower() == 'siglip2' or extractor.lower() == 'clip':
        img_inputs = img_preprocess(images=pil, return_tensors="pt").to(feature_extraction_model.device)
        img_feat = feature_extraction_model.get_image_features(**img_inputs)
    else:
        raise ValueError(f"Unsupported extractor: {extractor}")

    return img_feat

def store_top_k(cfg, task, query_ids, target_ids, description_feat, tar_tensor_feat, dataset_name, extractor, cutoff=50, **kwargs):
    target_ids = np.array(target_ids)
    sim = pairwise_cosine_similarity(description_feat, tar_tensor_feat)
    _, indices = sim.topk(k=cutoff, dim=1)
    predicted = [target_ids[row] for row in indices.tolist()]
    if dataset_name.lower() == 'circo':
        res={item[0]: item[1].astype(int).tolist() for item in zip(query_ids, predicted)}
    elif dataset_name.lower() == 'cirr':
        res={str(item[0]): item[1].tolist() for item in zip(query_ids, predicted)}
        res['version'] = 'rc2'
        res['metric'] = 'recall' if cutoff == 50 else 'recall_subset'
    if task == 'img2txt' or task == 'img2img':
        if kwargs.get('NUM_INFERENCE_STEPS'):
            num_inference_steps = kwargs['NUM_INFERENCE_STEPS']
        else:
            num_inference_steps = cfg['IMAGE-GENERATION']['GLOBAL']['NUM_INFERENCE_STEPS']
        if kwargs.get('GUIDANCE_SCALE'):
            guidance_scale = kwargs['GUIDANCE_SCALE']
        else:
            guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['GUIDANCE_SCALE']
        if kwargs.get('IMAGE_GUIDANCE_SCALE'):
            image_guidance_scale = kwargs['IMAGE_GUIDANCE_SCALE']
        else:
            image_guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['IMAGE_GUIDANCE_SCALE'] 
        json.dump(res, open(f"{task}_{dataset_name}_{extractor}_{num_inference_steps}_{image_guidance_scale}_{guidance_scale}_top{cutoff}_results.json", "w"))
    elif task == 'txt2img' or task == 'txt2txt':
        if kwargs.get('TEMPERATURE'):
            temperature= kwargs['TEMPERATURE']
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
        json.dump(res, open(f"{task}_{dataset_name}_{extractor}_{temperature}_{top_p}_{llm_top_k}_top{cutoff}_results.json", "w"))
        print('Predictions saved.')

def main(cfg, **kwargs):
    model_id = cfg['TEXT-GENERATION']['MODEL_NAME']
    top_k = cfg['GENERAL']['TOP_K']

    if kwargs.get('TASK'):
        task = kwargs['TASK']
    else:
        task = cfg['GENERAL']['TASK']
    if kwargs.get('DEVICE') is not None:
        device = torch.device(f"cuda:{kwargs['DEVICE']}")
    else:
        device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
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
    print(f"Using {model_id} for text generation with temperature={temperature}, top_p={top_p}, top_k={llm_top_k}, max_new_tokens={max_new_tokens}")

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
    print(f"Using {extractor} for feature extraction on {dataset_name} ({split})")

    feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, extractor=extractor)
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
                                                                               ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, 
                                              padding_side='left', 
                                              use_fast=True
                                              )
    dataloader = get_dataloader(cfg, 
                                split=split.lower(), 
                                dataset_name=dataset_name, 
                                extractor_name=extractor, 
                                batch_size=batch_size
                                )
    
    img_batch_size = 1024 if task == 'txt2img' else batch_size
    if dataset_name.lower() == 'cirr':
        if split.lower() == 'test1':
            test_loader = get_dataloader(cfg, dataset_name='cirr_target_image', batch_size=img_batch_size, extractor_name=extractor)
        elif split.lower() == 'train' or split.lower() == 'val':
            test_loader = get_dataloader(cfg, dataset_name=dataset_name, split=split.lower(), batch_size=img_batch_size, extractor_name=extractor)
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
    elif dataset_name.lower() == 'circo':
        if split.lower() == 'test':
            test_loader = get_dataloader(cfg, dataset_name='circo_target_image', batch_size=img_batch_size, extractor_name=extractor)
        elif split.lower() == 'val':
            test_loader = get_dataloader(cfg, dataset_name=dataset_name, split='val', batch_size=img_batch_size, extractor_name=extractor)
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
    elif dataset_name.lower() == 'fashioniq':
        if split.lower() == 'val' or split.lower() == 'train':
            test_loader = get_dataloader(cfg, dataset_name=dataset_name, split=split.lower(), batch_size=img_batch_size, extractor_name=extractor)
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
            
    description_feat = []
    query_ids = []
    tar_tensor_feat = []
    target_ids = []
    target_length = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            reference_pil = batch['reference_pil']
            caption = batch['caption']
            if dataset_name.lower() != 'fashioniq':
                query_ids.extend(batch['query_id'])

            composed_messages = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))

            composed_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model)
            print(composed_descriptions)
            gen_feat = extract_text_features(composed_descriptions, extractor, tokenizer, feature_extraction_model)
            description_feat.append(gen_feat)

        if task == 'txt2img':
            del text_generation_model
            gc.collect()
            torch.cuda.empty_cache()

            for j, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['all_target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['all_target_length'])

                tar_tensor_feat.append(extract_image_features(pil, extractor, feature_extraction_model, img_preprocess))
                
        elif task == 'txt2txt':
            for p, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['all_target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['all_target_length'])
                target_messages = list(map(lambda x: get_target_prompts(dataset_name, x), pil))
                target_description = generate_texts(target_messages, gen_config, processor, text_generation_model)
                print(target_description)
                tar_tensor_feat.append(extract_text_features(target_description, extractor, tokenizer, feature_extraction_model))

            del text_generation_model
            del feature_extraction_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unsupported task: {task}")
            
    description_feat = torch.cat(description_feat, dim=0)  
    tar_tensor_feat = torch.cat(tar_tensor_feat, dim=0)

    if dataset_name.lower() == 'circo' and split.lower() == 'test':
        store_top_k(cfg, task, query_ids, target_ids, description_feat, tar_tensor_feat, dataset_name, extractor, **kwargs)
    elif dataset_name.lower() == 'cirr' and split.lower() == 'test1':
        store_top_k(cfg, task, query_ids, target_ids, description_feat, tar_tensor_feat, dataset_name, extractor, **kwargs)
        store_top_k(cfg, task, query_ids, target_ids, description_feat, tar_tensor_feat, dataset_name, extractor, cutoff=30, **kwargs)
    elif dataset_name.lower() == 'fashioniq' and split.lower() == 'val':
        for k in top_k:
            fashioniq_eval(dataloader, description_feat, tar_tensor_feat, target_length, k)
    else:
        metric = 'map' if dataset_name.lower() == 'circo' else 'recall'
        for k in top_k:
            metric_val = get_metrics(description_feat,
                                      tar_tensor_feat,
                                      k=k,
                                      target_length=target_length,
                                      metrics=metric
                                    )
            print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated description ---> target images\n')

    print(f"{'*'*20}Completed{'*'*20}")
def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    set_seed(cfg['GENERAL']['SEED'])
    main(cfg, **kwargs)

if __name__ == "__main__":
    fire.Fire(launch)