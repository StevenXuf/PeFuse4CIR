import torch
import fire
import numpy as np
import json
import gc

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from configuration import get_default_config
from feature_extraction import get_feature_extractor
from dataloaders import get_dataloader

from text_generation_val import generate_target_description

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
                                                                               ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, 
                                              padding_side='left', 
                                              use_fast=True
                                              )
    split = 'test1' if dataset_name.lower() == 'cirr' else 'test'
    dataloader = get_dataloader(cfg, split=split, dataset_name=dataset_name, extractor_name=extractor, batch_size=batch_size)

    # caption_feat = []
    # modification_feat = []
    description_feat = []
    query_ids = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            reference_pil = batch['reference_pil']
            caption = batch['caption']
            query_ids.extend(batch['query_id'])

            # text_modification = list(map(lambda x: generate_text_modification(*x),zip(target_pil, reference_pil)))
            target_description = list(map(lambda x: generate_target_description(*x),zip(reference_pil, caption)))

            generated_text = []
            for text_info in [target_description]:
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
                generated_ids = text_generation_model.generate(**inputs,
                                                               generation_config=gen_config
                                                               ).detach()
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
            elif extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
                all_inputs = tokenizer(caption+generated_text,
                                       context_length=feature_extraction_model.context_length
                                       ).to(device)
                gen_feat = feature_extraction_model.encode_text(all_inputs)
            else:
                raise ValueError(f"Unsupported extractor: {extractor}")

            steps = gen_feat.size(0)//2
            # caption_feat.append(gen_feat[:steps, :])
            # modification_feat.append(gen_feat[steps:steps*2, :])
            description_feat.append(gen_feat[steps:, :])

            # if i == 1:
            #     break

    #clean generative models to save gpu memory
    del text_generation_model
    gc.collect()
    torch.cuda.empty_cache()

    if dataset_name.lower() == 'circo':
        test_loader = get_dataloader(cfg, dataset_name='circo_target_image', batch_size=512, extractor_name=extractor)
    elif dataset_name.lower() == 'cirr':
        test_loader = get_dataloader(cfg, dataset_name='cirr_target_image', batch_size=512, extractor_name=extractor)

    tar_tensor_feat = []
    target_ids=[]
    with torch.no_grad():
        for j, test_batch in tqdm(enumerate(test_loader)):
            image = test_batch['image']
            pil = test_batch['image_pil']
            target_ids.extend(test_batch['image_id'])
            if extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
                img_feat = feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in pil],dim=0).to(device))
            else:
                img_feat = feature_extraction_model.get_image_features(pixel_values=image.to(device))
            tar_tensor_feat.append(img_feat)

            # if j == 1:
            #     break

    # caption_feat = torch.cat(caption_feat, dim=0)
    # modification_feat = torch.cat(modification_feat, dim=0)
    description_feat = torch.cat(description_feat, dim=0)
    tar_tensor_feat = torch.cat(tar_tensor_feat, dim=0)

    target_ids = np.array(target_ids)
    sim = pairwise_cosine_similarity(description_feat, tar_tensor_feat)
    cutoff = 50
    _, indices = sim.topk(k=cutoff, dim=1)
    predicted = [target_ids[row] for row in indices.tolist()]
    if dataset_name.lower() == 'circo':
        res={item[0]: item[1].astype(int).tolist() for item in zip(query_ids, predicted)}
    elif dataset_name.lower() == 'cirr':
        res={str(item[0]): item[1].tolist() for item in zip(query_ids, predicted)}
        res['version'] = 'rc2'
        res['metric'] = 'recall' if cutoff == 50 else 'recall_subset'
    json.dump(res, open(f"predicted_results_txt_gen_{dataset_name}_{extractor}.json", "w"))

def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    main(cfg, **kwargs)

if __name__ == "__main__":
    fire.Fire(launch)