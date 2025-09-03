import torch
import os
import fire
import gc

from tqdm import tqdm
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig, set_seed
from qwen_vl_utils import process_vision_info

from figures import show_tensor_images
from configuration import get_default_config
from feature_extraction import get_metrics, get_feature_extractor
from dataloaders import get_dataloader
from text_to_image_and_text import fashioniq_eval, generate_texts, extract_text_features, extract_image_features, store_top_k
from prompts import get_composed_prompts, get_target_prompts
from utils import convert_pil_to_tensor, resize_crop_normalize, transform_image

def main(cfg, **kwargs):
    image_size = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['IMAGE_SIZE']
    top_k = cfg['GENERAL']['TOP_K']
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
    if kwargs.get('TASK'):
        task = kwargs['TASK']
    else:
        task = cfg['GENERAL']['TASK']
    if kwargs.get('DEVICE'):
        device = kwargs['DEVICE']
    else:
        device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
    if kwargs.get('EXTRACTOR'):
        extractor_name = kwargs['EXTRACTOR']
    else:
        extractor_name = cfg['GENERAL']['EXTRACTOR']
    if kwargs.get('EXTRACTOR_ID'):
        extractor_id = kwargs['EXTRACTOR_ID']
    else:
        extractor_id = None
    if kwargs.get('PRETRAINED'):
        pretrained = kwargs['PRETRAINED']
    else:
        pretrained = cfg[extractor_name]['PRETRAINED']
    extractor_id = cfg[extractor_name]['MODEL_NAME']
    if kwargs.get('DATASET'):
        dataset_name = kwargs['DATASET']
    else:
        dataset_name = cfg['GENERAL']['DATASET']
    print(f"Using {extractor_name} with id: {extractor_id} for feature extraction on {dataset_name} dataset.")

    if kwargs.get('SPLIT'):
        split = kwargs['SPLIT']
    else:
        split = cfg['GENERAL']['SPLIT']
    if kwargs.get('NUM_INFERENCE_STEPS'):
        n_infer_step = kwargs['NUM_INFERENCE_STEPS']
    else:
        n_infer_step = cfg['IMAGE-GENERATION']['GLOBAL']['NUM_INFERENCE_STEPS']
    if kwargs.get('IMAGE_GUIDANCE_SCALE'):
        image_guidance_scale = kwargs['IMAGE_GUIDANCE_SCALE']
    else:
        image_guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['IMAGE_GUIDANCE_SCALE']
    if kwargs.get('GUIDANCE_SCALE'):
        guidance_scale = kwargs['GUIDANCE_SCALE']
    else:
        guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['GUIDANCE_SCALE']
    
    if kwargs.get('BATCH_SIZE'):
        batch_size = kwargs['BATCH_SIZE']
    else:
        batch_size = cfg['GENERAL']['BATCH_SIZE']

    if kwargs.get('use_llm'):
        use_llm = kwargs['use_llm']
    else:
        use_llm = cfg['GENERAL']['use_llm']

    img_transform_for_generation = transform_image(image_size)
    dataloader = get_dataloader(cfg, 
                                split=split.lower(),
                                batch_size=batch_size,
                                mode='relative',
                                transform=img_transform_for_generation, 
                                extractor_name=extractor_name,
                                dataset_name=dataset_name)

    img_transform_for_extraction = transform_image(cfg[extractor_name]['IMAGE_SIZE'], 
                                                   cfg[extractor_name]['IMAGE_MEAN'], 
                                                   cfg[extractor_name]['IMAGE_STD']
                                                   )
    
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

    feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, extractor=extractor_name, extractor_id=extractor_id, pretrained=pretrained)
    feature_extraction_model.eval()
    feature_extraction_model.to(device)

    img_batch_size = 1024 if task == 'img2img' else batch_size
    if dataset_name.lower() == 'cirr':
        if split.lower() == 'test':
            test_loader = get_dataloader(cfg, 
                                         mode='classic',
                                         dataset_name=dataset_name, 
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor_name)
        elif split.lower() == 'train' or split.lower() == 'val':
            test_loader = get_dataloader(cfg, 
                                         dataset_name=dataset_name, 
                                         split=split.lower(),
                                         mode='relative',
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor_name
                                         )
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
    elif dataset_name.lower() == 'circo':
        if split.lower() == 'test':
            test_loader = get_dataloader(cfg, 
                                         mode='classic',
                                         dataset_name=dataset_name, 
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor_name)
        elif split.lower() == 'val':
            test_loader = get_dataloader(cfg, 
                                         dataset_name=dataset_name, 
                                         split='val', 
                                         mode='relative',
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor_name
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
                                         extractor_name=extractor_name
                                         )
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")

    generator = torch.Generator(device="cuda").manual_seed(cfg['GENERAL']['SEED'])
    generation_model=StableDiffusionXLInstructPix2PixPipeline.from_pretrained(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['MODEL_NAME'], 
                                                                              torch_dtype=torch.float16).to(device)
    print(f"Using {generation_model.__class__.__name__} with params: n_infer_step={n_infer_step}, image_guidance_scale={image_guidance_scale}, guidance_scale={guidance_scale}")

    store_path = os.path.join(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['OUTPUT_DIR'], f'Qwen_{use_llm}_{dataset_name}_{extractor_name}_{n_infer_step}_{image_guidance_scale}_{guidance_scale}')
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    generated_target_features = []
    target_features = []
    target_length = []
    target_ids = []
    query_ids = []
    fashioniq_ground_truth = []
    if dataset_name.lower() == 'cirr' and split.lower() == 'test':
        img_subset = []
        img_subset_ids = []
        img_subset_feat = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i,batch in tqdm(enumerate(dataloader)):
            input_images = batch['reference_img']
            reference_pil = batch['reference_pil']
            query_ids.extend(batch['query_id'])

            if dataset_name.lower() == 'fashioniq':
                fashioniq_ground_truth.extend(batch['target_id'])
            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                img_subset.extend(batch['image_set'])
                img_subset_ids.extend(batch['image_subset_ids'])
            caption = batch['caption']
            if use_llm.lower() == 'yes':
                composed_messages = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))
                caption = generate_texts(composed_messages, gen_config, processor, text_generation_model)
                print(caption)

            show_tensor_images(input_images, num_images=input_images.size(0), file_path=os.path.join(store_path,f"reference_image_grid_{i}.png"))
            # show_tensor_images(targets, num_images=targets.size(0), file_path=os.path.join(store_path,f"target_image_grid_{j}.png"))

            print(f"Generating target images for batch {i+1}")
            generated_target_images = generation_model(
                prompt=caption,
                image=input_images.to(device),
                width=image_size,
                height=image_size,
                num_inference_steps=n_infer_step,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
                generator=generator
                ).images
            generated_target_image_tensor = torch.stack(convert_pil_to_tensor(generated_target_images))
            show_tensor_images(generated_target_image_tensor, num_images=generated_target_image_tensor.size(0), file_path=os.path.join(store_path,f"generated_target_image_grid_{i}.png"))

            generated_target_features.append(extract_image_features(generated_target_images, extractor_name, feature_extraction_model, img_preprocess))
            print(f'Batch {i+1} finished.')

        if task == 'img2img':
            
            for j, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])

                target_features.append(extract_image_features(pil, extractor_name, feature_extraction_model, img_preprocess))
            print("Finished extracting target image features")

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//img_batch_size + 1 if len(img_subset)%img_batch_size != 0 else len(img_subset)//img_batch_size
                for i in range(n_iters):
                    img_subset_feat.append(extract_image_features(img_subset[i*img_batch_size:(i+1)*img_batch_size], extractor_name, feature_extraction_model, img_preprocess))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")

        elif task == 'image2txt':
            for p, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])
                target_messages = list(map(lambda x: get_target_prompts(dataset_name, x), pil))
                target_description = generate_texts(target_messages, gen_config, processor, text_generation_model)
                print(target_description)
                target_features.append(extract_text_features(target_description, extractor_name, tokenizer, feature_extraction_model))
            print("Finished generating target descriptions and extracting features")

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//32 + 1 if len(img_subset)%32 != 0 else len(img_subset)//32
                for i in range(n_iters):
                    target_messages = list(map(lambda x: get_target_prompts(dataset_name, x), img_subset[i*32:(i+1)*32]))
                    target_description = generate_texts(target_messages, gen_config, processor, text_generation_model)
                    img_subset_feat.append(extract_text_features(target_description, extractor_name, tokenizer, feature_extraction_model))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")

        else:
            raise ValueError(f"Unknown task: {task}")

    generated_target_features = torch.cat(generated_target_features, dim=0)
    target_features = torch.cat(target_features, dim=0)

    if dataset_name.lower() == 'circo' and split.lower() == 'test':
        store_top_k(cfg, task, query_ids, target_ids, generated_target_features, target_features, dataset_name, extractor_name, **kwargs)
    elif dataset_name.lower() == 'cirr' and split.lower() == 'test':
        store_top_k(cfg, task, query_ids, target_ids, generated_target_features, target_features, dataset_name, extractor_name, **kwargs)
        store_top_k(cfg, task, query_ids, img_subset_ids, generated_target_features, img_subset_feat, dataset_name, extractor_name, cutoff=3, **kwargs)
    elif dataset_name.lower() == 'fashioniq' and split.lower() == 'val':
        for k in top_k:
            fashioniq_eval(dataloader, generated_target_features, target_features, fashioniq_ground_truth, target_ids, k)
    else:
        metric = 'map' if dataset_name.lower() == 'circo' else 'recall' ######modify here!!!!!
        for k in top_k:
            metric_val = get_metrics(generated_target_features,
                                      target_features,
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