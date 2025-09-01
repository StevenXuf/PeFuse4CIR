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

def fashioniq_eval(dataloader, generated_target_features, target_features, ground_truth, target_id, k):
    clothes = ['shirt', 'dress', 'toptee']
    truth_start = 0
    tar_start = 0
    all_recall = []
    for j, (n, c) in enumerate(zip(dataloader.dataset.length, dataloader.dataset.candidate_length)):
        vals, ids = pairwise_cosine_similarity(generated_target_features[truth_start:truth_start+n, :], target_features[tar_start:tar_start+c, :]).topk(k=k, dim=1)
        total_recall=0.0
        for i in range(n):
            truth = ground_truth[truth_start:truth_start+n][i]
            preds = [target_id[tar_start:tar_start+c][idx] for idx in ids[i].tolist()]
            if truth in preds:
                total_recall += 1.0
        print(f"Recall@{k}: {total_recall / n * 100:.2f}% for {clothes[j]}")
        all_recall.append(total_recall / n * 100)
        truth_start += n
        tar_start += c

    print(f'Recall@{k}: {np.mean(all_recall):.2f}% for all categories')

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
    elif extractor.lower() == 'clip':
        img_inputs = torch.cat([img_preprocess(img).unsqueeze(0) for img in pil],dim=0).to(feature_extraction_model.device)
        img_feat = feature_extraction_model.get_image_features(pixel_values=img_inputs)
    elif extractor.lower() == 'siglip2':
        img_inputs = img_preprocess(images=pil, return_tensors="pt").to(feature_extraction_model.device)
        img_feat = feature_extraction_model.get_image_features(**img_inputs)
    else:
        raise ValueError(f"Unsupported extractor: {extractor}")

    return img_feat

def store_top_k(cfg, task, query_ids, target_ids, description_feat, tar_tensor_feat, dataset_name, extractor, cutoff=50, **kwargs):
    if cutoff == 3:
        start=0
        res = {'version': 'rc2', 'metric': 'recall_subset'}
        for i in range(len(query_ids)):
            current_ids = target_ids[i]
            _,ids = pairwise_cosine_similarity(description_feat[i,:], tar_tensor_feat[start:start+len(current_ids), :]).topk(k=cutoff, dim=1)
            res[str(query_ids[i])] = [current_ids[idx] for idx in ids[0].tolist()]
            start += len(current_ids)
    else:
        target_ids = np.array(target_ids)
        sim = pairwise_cosine_similarity(description_feat, tar_tensor_feat)
        _, indices = sim.topk(k=cutoff, dim=1)
        predicted = [target_ids[row] for row in indices.tolist()]
        if dataset_name.lower() == 'circo':
            res={item[0]: item[1].astype(int).tolist() for item in zip(query_ids, predicted)}
        elif dataset_name.lower() == 'cirr':
            res={str(item[0]): item[1].tolist() for item in zip(query_ids, predicted)}
            res['version'] = 'rc2'
            res['metric'] = 'recall'
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
    if kwargs.get('EXTRACTOR_ID'):
        extractor_id = kwargs['EXTRACTOR_ID']
    else:
        extractor_id = None
    if kwargs.get('PRETRAINED'):
        pretrained = kwargs['PRETRAINED']
    else:
        pretrained = cfg[extractor]['PRETRAINED']
    if kwargs.get('DATASET'):
        dataset_name = kwargs['DATASET']
    else:
        dataset_name = cfg['GENERAL']['DATASET']
    if kwargs.get('SPLIT'):
        split = kwargs['SPLIT']
    else:
        split = cfg['GENERAL']['SPLIT']
    if kwargs.get('MODE'):
        mode = kwargs['MODE']
    else:
        mode = cfg['GENERAL']['MODE']
    print(f"Using {extractor} for feature extraction on {dataset_name} ({split})")

    feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, extractor=extractor, extractor_id=extractor_id, pretrained=pretrained)
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
                                mode='relative',
                                dataset_name=dataset_name, 
                                extractor_name=extractor,
                                batch_size=batch_size
                                )

    img_batch_size = 1024 if task == 'txt2img' else batch_size
    if dataset_name.lower() == 'cirr':
        if split.lower() == 'test':
            test_loader = get_dataloader(cfg, 
                                         mode='classic',
                                         dataset_name=dataset_name, 
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor)
        elif split.lower() == 'train' or split.lower() == 'val':
            test_loader = get_dataloader(cfg, 
                                         dataset_name=dataset_name, 
                                         split=split.lower(),
                                         mode='relative',
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor
                                         )
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
    elif dataset_name.lower() == 'circo':
        if split.lower() == 'test':
            test_loader = get_dataloader(cfg, 
                                         mode='classic',
                                         dataset_name=dataset_name, 
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor)
        elif split.lower() == 'val':
            test_loader = get_dataloader(cfg, 
                                         dataset_name=dataset_name, 
                                         split='val', 
                                         mode='relative',
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor
                                         )
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
    elif dataset_name.lower() == 'fashioniq':
        if split.lower() == 'val' or split.lower() == 'train':
            test_loader = get_dataloader(cfg, 
                                         dataset_name=dataset_name, 
                                         split=split.lower(), 
                                         mode='classic',
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor,
                                         )
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
            
    description_feat = []
    query_ids = []
    tar_tensor_feat = []
    target_ids = []
    target_length = []
    fashioniq_ground_truth = []
    if dataset_name.lower() == 'cirr' and split.lower() == 'test':
        img_subset = []
        img_subset_ids = []
        img_subset_feat = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            reference_pil = batch['reference_pil']
            caption = batch['caption']
            query_ids.extend(batch['query_id'])
            if dataset_name.lower() == 'fashioniq':
                fashioniq_ground_truth.extend(batch['target_id'])
            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                img_subset.extend(batch['image_set'])
                img_subset_ids.extend(batch['image_subset_ids'])

            composed_messages = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))

            composed_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model)
            print(composed_descriptions)
            gen_feat = extract_text_features(composed_descriptions, extractor, tokenizer, feature_extraction_model)
            description_feat.append(gen_feat)
        print("Finished generating composed descriptions")

        if task == 'txt2img':
            del text_generation_model
            gc.collect()
            torch.cuda.empty_cache()

            for j, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])

                tar_tensor_feat.append(extract_image_features(pil, extractor, feature_extraction_model, img_preprocess))
            print("Finished extracting target image features") 

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//img_batch_size + 1 if len(img_subset)%img_batch_size != 0 else len(img_subset)//img_batch_size
                for i in range(n_iters):
                    img_subset_feat.append(extract_image_features(img_subset[i*img_batch_size:(i+1)*img_batch_size], extractor, feature_extraction_model, img_preprocess))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")

        elif task == 'txt2txt':
            for p, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])
                target_messages = list(map(lambda x: get_target_prompts(dataset_name, x), pil))
                target_description = generate_texts(target_messages, gen_config, processor, text_generation_model)
                print(target_description)
                tar_tensor_feat.append(extract_text_features(target_description, extractor, tokenizer, feature_extraction_model))
            print("Finished generating target descriptions and extracting features")

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//32 + 1 if len(img_subset)%32 != 0 else len(img_subset)//32
                for i in range(n_iters):
                    target_messages = list(map(lambda x: get_target_prompts(dataset_name, x), img_subset[i*32:(i+1)*32]))
                    target_description = generate_texts(target_messages, gen_config, processor, text_generation_model)
                    img_subset_feat.append(extract_text_features(target_description, extractor, tokenizer, feature_extraction_model))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")
            
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
    elif dataset_name.lower() == 'cirr' and split.lower() == 'test':
        store_top_k(cfg, task, query_ids, target_ids, description_feat, tar_tensor_feat, dataset_name, extractor, **kwargs)
        store_top_k(cfg, task, query_ids, img_subset_ids, description_feat, img_subset_feat, dataset_name, extractor, cutoff=3, **kwargs)
    elif dataset_name.lower() == 'fashioniq' and split.lower() == 'val':
        for k in top_k:
            fashioniq_eval(dataloader, description_feat, tar_tensor_feat, fashioniq_ground_truth, target_ids, k)
    else:
        metric = 'map' if dataset_name.lower() == 'circo' else 'recall' ######modify here!!!!!
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


####use targetpad to improve CLIP/OpenCLIP
####Adjust the prompts