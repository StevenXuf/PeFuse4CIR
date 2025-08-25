import torch
import fire

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info

from configuration import get_default_config
from feature_extraction import get_metrics, get_feature_extractor
from dataloaders import get_dataloader
from prompts import get_composed_prompts

def fashioniq_eval(dataloader, generated_target_features, target_features, target_length, k):
    n_shirt, n_dress, n_toptee = dataloader.dataset.length
    shirt_metric_val = get_metrics(generated_target_features[:n_shirt, :], target_features[:n_shirt, :], k=k, target_length=target_length[:n_shirt], metrics='recall')
    dress_metric_val = get_metrics(generated_target_features[n_shirt:n_shirt+n_dress, :], target_features[n_shirt:n_shirt+n_dress, :], k=k, target_length=target_length[n_shirt:n_shirt+n_dress], metrics='recall')
    toptee_metric_val = get_metrics(generated_target_features[n_shirt+n_dress:, :], target_features[n_shirt+n_dress:, :], k=k, target_length=target_length[n_shirt+n_dress:], metrics='recall')
    print(f'Recall@{k}: {shirt_metric_val:.2f}% for shirt')
    print(f'Recall@{k}: {dress_metric_val:.2f}% for dress')
    print(f'Recall@{k}: {toptee_metric_val:.2f}% for toptee')
    print(f'Recall@{k}: {(shirt_metric_val+dress_metric_val+toptee_metric_val)/3:.2f}% for all categories')

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
    
    if kwargs.get('EXTRACTOR'):
        extractor = kwargs['EXTRACTOR']
    else:
        extractor = cfg['GENERAL']['EXTRACTOR']
    if kwargs.get('DATASET'):
        dataset_name = kwargs['DATASET']
    else:
        dataset_name = cfg['GENERAL']['DATASET']
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

    if kwargs.get('SPLIT'):
        split = kwargs['SPLIT']
    else:
        split = cfg['GENERAL']['SPLIT']
    if kwargs.get('BATCH_SIZE'):
        batch_size = kwargs['BATCH_SIZE']
    else:
        batch_size = cfg['GENERAL']['BATCH_SIZE']
    dataloader = get_dataloader(cfg, 
                                split=split, 
                                extractor_name=extractor, 
                                batch_size=batch_size,
                                dataset_name=dataset_name)

    # caption_feat = []
    # modification_feat = []
    description_feat = []
    tar_tensor_feat = []
    target_length = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i, batch in tqdm(enumerate(dataloader), desc="Gnerating descriptions", total=len(dataloader)):
            target_pil = batch['target_pil']
            reference_pil = batch['reference_pil']
            # reference_tensor = batch['reference_img']
            caption = batch['caption']
            # target_tensor = batch['target_img']
            all_target_pil = batch['all_target_pil']
            all_target_tensor = batch['all_target_img']
            target_length.extend(batch['all_target_length'])

            # text_modification = list(map(lambda x: generate_text_modification(*x),zip(target_pil, reference_pil)))
            composed_description = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))

            generated_text = []
            for text_info in [composed_description]:
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
                    padding_side="left"
                )
                inputs = inputs.to(device)

                # Batch Inference
                generated_ids = text_generation_model.generate(**inputs,
                                                               generation_config=gen_config,
                                                               )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                generated_text.extend(output_texts)
                print(output_texts)

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
            if extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
                img_feat = feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in all_target_pil],dim=0).to(device))
            else:
                img_feat = feature_extraction_model.get_image_features(pixel_values=all_target_tensor.to(device))
            tar_tensor_feat.append(img_feat)

    # print(target_length)
    # caption_feat = torch.cat(caption_feat, dim=0)
    # modification_feat = torch.cat(modification_feat, dim=0)
    description_feat = torch.cat(description_feat, dim=0)
    tar_tensor_feat = torch.cat(tar_tensor_feat, dim=0)

    if dataset_name.lower() == "circo":
        metric = 'map'
    else:
        metric = 'recall'
    for k in top_k:
        #compute recall for generated modification ---> caption
        # metric_val = get_metrics(modification_feat, caption_feat, k=k, target_length=target_length, metrics=metric)
        # print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated modification ---> real modification')

        if dataset_name.lower() == 'fashioniq' and split == 'val':
            fashioniq_eval(dataloader, description_feat, tar_tensor_feat, target_length, k)
        #compute recall for generated description ---> target image
        else:
            metric_val = get_metrics(description_feat, tar_tensor_feat, k=k, target_length=target_length, metrics=metric)
            print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated description ---> target images\n')
    print(f"{'*'*20}Text generation completed{'*'*20}")

def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    main(cfg, **kwargs)

if __name__ == "__main__":
    fire.Fire(launch)

    #### TO DO: MODIFY BACKWARD COMPUTATION FOR text and image gen ####
    ### Use image-only, text-only, or combined as baselines
    ### Check the order of the reference images and the target images!!!!!!!