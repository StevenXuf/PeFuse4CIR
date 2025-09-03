import torch
import fire
import os

from tqdm import tqdm
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig, set_seed

from figures import show_tensor_images
from configuration import get_default_config
from feature_extraction import get_feature_extractor, get_metrics
from dataloaders import get_dataloader
from text_to_image_and_text import fashioniq_eval, generate_texts, extract_text_features, extract_image_features, store_top_k
from prompts import get_composed_prompts, get_target_prompts
from utils import convert_pil_to_tensor, resize_crop_normalize, transform_image, delete_models

def main(cfg, **kwargs):
    ### General Parameters
    top_k = cfg['GENERAL']['TOP_K']
    if kwargs.get('TASK'):
        task = kwargs['TASK']
    else:
        task = cfg['GENERAL']['TASK']
    if kwargs.get('DEVICE') is not None:
        device = torch.device(f"cuda:{kwargs['DEVICE']}")
    else:
        device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
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
    print(f"Using {extractor} for feature extraction on {dataset_name} ({split})")

    ### Text Generation Parameters
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

    ### Image Generation Parameters
    image_size = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['IMAGE_SIZE']
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
    if kwargs.get('USE_LLM'):
        use_llm = kwargs['USE_LLM']
    else:
        use_llm = cfg['GENERAL']['USE_LLM']

    feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, 
                                                                                extractor=extractor, 
                                                                                extractor_id=extractor_id, 
                                                                                pretrained=pretrained
                                                                                )
    feature_extraction_model.eval()
    feature_extraction_model.to(device)

    if task.startswith('img2'):
        img_transform_for_generation = transform_image(image_size)
        generator = torch.Generator(device="cuda").manual_seed(cfg['GENERAL']['SEED'])
        image_generation_model=StableDiffusionXLInstructPix2PixPipeline.from_pretrained(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['MODEL_NAME'], 
                                                                                  torch_dtype=torch.float16).to(device)
        print(f"Using {image_generation_model.__class__.__name__} with params: n_infer_step={n_infer_step}, image_guidance_scale={image_guidance_scale}, guidance_scale={guidance_scale}")

        store_path = os.path.join(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['OUTPUT_DIR'], f'Qwen_{use_llm}_{dataset_name}_{extractor}_{n_infer_step}_{image_guidance_scale}_{guidance_scale}')
        if not os.path.exists(store_path):
            os.makedirs(store_path)

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
                                batch_size=batch_size,
                                transform=img_transform_for_generation if task.startswith('img2') else None
                                )
    
    img_batch_size = 1024 if task.endswith('2img') else batch_size
    if dataset_name.lower() == 'cirr':
        if split.lower() == 'test':
            test_loader = get_dataloader(cfg, 
                                         mode='classic',
                                         dataset_name=dataset_name, 
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor
                                         )
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
            
    query_feat = []
    query_ids = []
    target_feat = []
    target_ids = []
    target_length = []
    fashioniq_ground_truth = []
    if dataset_name.lower() == 'cirr' and split.lower() == 'test':
        img_subset = []
        img_subset_ids = []
        img_subset_feat = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i, batch in tqdm(enumerate(dataloader), desc=f"Extracting query features for {task}", total=len(dataloader)):
            reference_pil = batch['reference_pil']
            caption = batch['caption']
            query_ids.extend(batch['query_id'])
            if dataset_name.lower() == 'fashioniq':
                fashioniq_ground_truth.extend(batch['target_id'])
            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                img_subset.extend(batch['image_set'])
                img_subset_ids.extend(batch['image_subset_ids'])

            if task.startswith('img2'):
                reference_img = batch['reference_img']
                if use_llm == 'yes':
                    composed_messages = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))
                    composed_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model)
                    print(composed_descriptions)

                show_tensor_images(reference_img, num_images=reference_img.size(0), file_path=os.path.join(store_path,f"reference_image_grid_{i}.png"))
                print(f"Generating target images for batch {i+1}")
                generated_target_images = image_generation_model(
                    prompt=composed_descriptions,
                    image=reference_img.to(device),
                    width=image_size,
                    height=image_size,
                    num_inference_steps=n_infer_step,
                    image_guidance_scale=image_guidance_scale,
                    guidance_scale=guidance_scale,
                    generator=generator
                    ).images
                generated_target_image_tensor = torch.stack(convert_pil_to_tensor(generated_target_images))
                show_tensor_images(generated_target_image_tensor, num_images=generated_target_image_tensor.size(0), file_path=os.path.join(store_path,f"generated_target_image_grid_{i}.png"))

                query_feat.append(extract_image_features(generated_target_images, extractor, feature_extraction_model, img_preprocess))
                print(f'Batch {i+1} finished.')

            elif task.startswith('txt2'):
                composed_messages = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))
                composed_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model)
                print(composed_descriptions)
                query_feat.append(extract_text_features(composed_descriptions, extractor, tokenizer, feature_extraction_model))
            else:
                raise ValueError(f"Unsupported task: {task}. Should be one of ['txt2img', 'txt2txt', 'img2img', 'img2txt']")

        print(f"Finished extracting query features for {task} on {dataset_name}.".upper())

        if task.endswith('2img'):
            delete_models(text_generation_model)
            if task == 'img2img':
                delete_models(image_generation_model)

            for j, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])

                target_feat.append(extract_image_features(pil, extractor, feature_extraction_model, img_preprocess))
            print("Finished extracting target image features") 

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//img_batch_size + 1 if len(img_subset)%img_batch_size != 0 else len(img_subset)//img_batch_size
                for i in range(n_iters):
                    img_subset_feat.append(extract_image_features(img_subset[i*img_batch_size:(i+1)*img_batch_size], extractor, feature_extraction_model, img_preprocess))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")

        elif task.endswith('2txt'):
            if task == 'img2txt':
                delete_models(image_generation_model)
            for p, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])
                target_messages = list(map(lambda x: get_target_prompts(dataset_name, x), pil))
                target_description = generate_texts(target_messages, gen_config, processor, text_generation_model)
                print(target_description)
                target_feat.append(extract_text_features(target_description, extractor, tokenizer, feature_extraction_model))
            print("Finished generating target descriptions and extracting features")

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//32 + 1 if len(img_subset)%32 != 0 else len(img_subset)//32
                for i in range(n_iters):
                    target_messages = list(map(lambda x: get_target_prompts(dataset_name, x), img_subset[i*32:(i+1)*32]))
                    target_description = generate_texts(target_messages, gen_config, processor, text_generation_model)
                    img_subset_feat.append(extract_text_features(target_description, extractor, tokenizer, feature_extraction_model))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")
            
            delete_models(feature_extraction_model, text_generation_model)
        else:
            raise ValueError(f"Unsupported task: {task}")
            
    query_feat = torch.cat(query_feat, dim=0)  
    target_feat = torch.cat(target_feat, dim=0)

    if dataset_name.lower() == 'circo' and split.lower() == 'test':
        store_top_k(cfg, task, query_ids, target_ids, query_feat, target_feat, dataset_name, extractor, **kwargs)
    elif dataset_name.lower() == 'cirr' and split.lower() == 'test':
        store_top_k(cfg, task, query_ids, target_ids, query_feat, target_feat, dataset_name, extractor, **kwargs)
        store_top_k(cfg, task, query_ids, img_subset_ids, query_feat, img_subset_feat, dataset_name, extractor, cutoff=3, **kwargs)
    elif dataset_name.lower() == 'fashioniq' and split.lower() == 'val':
        for k in top_k:
            fashioniq_eval(dataloader, query_feat, target_feat, fashioniq_ground_truth, target_ids, k)
    else:
        metric = 'map' if dataset_name.lower() == 'circo' else 'recall' ######modify here!!!!!
        for k in top_k:
            metric_val = get_metrics(query_feat,
                                      target_feat,
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