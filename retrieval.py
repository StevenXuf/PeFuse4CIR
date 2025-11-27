import torch
import fire
import os
import time
import numpy as np

from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLInstructPix2PixPipeline, AutoPipelineForImage2Image, AutoPipelineForText2Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig, set_seed

from figures import show_tensor_images
from feature_extraction import get_feature_extractor, get_metrics
from dataloaders import get_dataloader
from text_to_image_and_text import fashioniq_eval, genecis_eval, generate_texts, extract_text_features, extract_image_features, store_top_k
from prompts import get_composed_prompts, get_target_prompts
from utils import get_default_config, convert_pil_to_tensor, transform_image, delete_models, get_gpu_memory


def get_test_loader(cfg, dataset_name, split, extractor, img_batch_size):
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
                                         extractor_name=extractor
                                         )
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
    elif dataset_name.lower() == 'change_attribute' or dataset_name.lower() == 'focus_attribute' or dataset_name.lower() == 'change_object' or dataset_name.lower() == 'focus_object':
        test_loader = get_dataloader(
            cfg,
            dataset_name=dataset_name,
            extractor_name=extractor,
            batch_size=img_batch_size
        )
    return test_loader

def main(cfg, **kwargs):
    ### General Parameters
    top_k = kwargs.get('top_k', cfg['GENERAL']['TOP_K'])
    task = kwargs.get('task', cfg['GENERAL']['TASK'])
    device = torch.device(f"cuda:{kwargs['device']}") if kwargs.get('device') is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = kwargs.get('batch_size', cfg['GENERAL']['BATCH_SIZE'])
    extractor = kwargs.get('extractor', cfg['GENERAL']['EXTRACTOR'])
    extractor_id = kwargs.get('extractor_id', None)
    pretrained = kwargs.get('pretrained', cfg[extractor]['PRETRAINED'])
    dataset_name = kwargs.get('dataset', cfg['GENERAL']['DATASET'])
    split = kwargs.get('split', cfg['GENERAL']['SPLIT'])
    print(f"Using {extractor} for feature extraction on {dataset_name} ({split})")

    ### Text Generation Parameters
    model_id = cfg['TEXT-GENERATION']['QWEN']['MODEL_NAME']
    temperature = kwargs.get('temperature', cfg['TEXT-GENERATION']['GLOBAL']['TEMPERATURE'])
    top_p = kwargs.get('top_p', cfg['TEXT-GENERATION']['GLOBAL']['TOP_P'])
    llm_top_k = kwargs.get('llm_top_k', cfg['TEXT-GENERATION']['GLOBAL']['TOP_K'])
    max_new_tokens = kwargs.get('max_new_tokens', cfg['TEXT-GENERATION']['GLOBAL']['MAX_NEW_TOKENS'])

    feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, 
                                                                                extractor=extractor, 
                                                                                extractor_id=extractor_id, 
                                                                                pretrained=pretrained
                                                                                )
    feature_extraction_model.to(device)
    feature_extraction_model.eval()

    if task.startswith('img2'):
        ### Image Generation Parameters
        seed = kwargs.get('seed', cfg['GENERAL']['SEED'])
        image_gen_model_name = kwargs.get('image_gen_model', cfg['GENERAL']['IMAGE_GEN_MODEL'])
        image_size = cfg['IMAGE-GENERATION'][image_gen_model_name]['IMAGE_SIZE']
        n_infer_steps = kwargs.get('n_infer_steps', cfg['IMAGE-GENERATION'][image_gen_model_name]['NUM_INFERENCE_STEPS'])
        image_guidance_scale = kwargs.get('image_guidance_scale', cfg['IMAGE-GENERATION'][image_gen_model_name]['IMAGE_GUIDANCE_SCALE'])
        guidance_scale = kwargs.get('guidance_scale', cfg['IMAGE-GENERATION'][image_gen_model_name]['GUIDANCE_SCALE'])
        use_llm = kwargs.get('use_llm', cfg['GENERAL']['USE_LLM'])

        img_transform_for_generation = transform_image(image_size)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        before = get_gpu_memory(device)
        if image_gen_model_name.upper() == 'SDXL-INSTRUCTPIX2PIX':
            image_generation_model = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(cfg['IMAGE-GENERATION'][image_gen_model_name]['MODEL_NAME'], 
                                                                                            torch_dtype=torch.float16).to(device)
        elif image_gen_model_name.upper() == 'SDXL-TURBO':
            pipeline_text2image = AutoPipelineForText2Image.from_pretrained(cfg['IMAGE-GENERATION'][image_gen_model_name]['MODEL_NAME'], 
                                                                            torch_dtype=torch.float16, 
                                                                            variant="fp16").to(device)
            image_generation_model = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)
        else:
            raise ValueError(f"Unsupported image generation model: {cfg['GENERAL']['IMAGE_GEN_MODEL']}")
        after = get_gpu_memory(device)
        print(f"Meomeroy taken by {image_gen_model_name}: {(after - before) / 1024**2: .2f} MB\n")
        image_generation_model.enable_xformers_memory_efficient_attention()
        image_generation_model.enable_attention_slicing()
        print(f"Using {image_generation_model.__class__.__name__} with params: n_infer_steps={n_infer_steps}, image_guidance_scale={image_guidance_scale}, guidance_scale={guidance_scale}")

        store_path = os.path.join(cfg['IMAGE-GENERATION'][image_gen_model_name]['OUTPUT_DIR'], f'Qwen_{use_llm}_{dataset_name}_{split}_{extractor}_{n_infer_steps}_{image_guidance_scale}_{guidance_scale}')
        if not os.path.exists(store_path):
            os.makedirs(store_path)

    gen_config = GenerationConfig(do_sample=True,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=llm_top_k,
                                    max_new_tokens=max_new_tokens
                                )
    before = get_gpu_memory(device)
    text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, 
                                                                            torch_dtype=torch.bfloat16, 
                                                                            device_map={"": device}, 
                                                                            # attn_implementation='flash_attention_2'
                                                                            ).to(device)
    after = get_gpu_memory(device)
    print(f"Meomeroy taken by {cfg['GENERAL']['MLLM']}: {(after - before) / 1024**2: .2f} MB\n")
    text_generation_model.eval()
    print(f"Using {text_generation_model.__class__.__name__} for text generation with temperature={temperature}, top_p={top_p}, top_k={llm_top_k}, max_new_tokens={max_new_tokens}")
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
    
    img_batch_size = 256 if task.endswith('2img') else batch_size

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
    with torch.no_grad():
        start_time_query = time.time()
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
                    caption = generate_texts(composed_messages, gen_config, processor, text_generation_model)
                    print(caption)

                show_tensor_images(reference_img, num_images=reference_img.size(0), file_path=os.path.join(store_path,f"reference_image_grid_{i}.png"))
                print(f"Generating target images for batch {i+1}")
                if image_gen_model_name.upper() == 'SDXL-INSTRUCTPIX2PIX':
                    generated_target_images = image_generation_model(
                        prompt=caption,
                        image=reference_img.to(device),
                        width=image_size,
                        height=image_size,
                        num_inference_steps=n_infer_steps,
                        image_guidance_scale=image_guidance_scale,
                        guidance_scale=guidance_scale,
                        generator=generator
                        ).images
                elif image_gen_model_name.upper() == 'SDXL-TURBO':
                    images = list(map(lambda x: x.resize((image_size, image_size)), reference_pil))
                    generated_target_images = image_generation_model(
                        prompt=caption,
                        image=images,
                        width=image_size,
                        height=image_size,
                        num_inference_steps=n_infer_steps,
                        strength=image_guidance_scale,
                        guidance_scale=guidance_scale,
                        generator=generator
                        ).images
                generated_target_image_tensor = torch.stack(convert_pil_to_tensor(generated_target_images))
                show_tensor_images(generated_target_image_tensor, num_images=generated_target_image_tensor.size(0), file_path=os.path.join(store_path,f"generated_target_image_grid_{i}.png"))

                query_feat.append(extract_image_features(generated_target_images, extractor, feature_extraction_model, img_preprocess))

            elif task.startswith('txt2'):
                composed_messages = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))
                composed_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model)
                print(composed_descriptions)
                query_feat.append(extract_text_features(composed_descriptions, extractor, tokenizer, feature_extraction_model))
            else:
                raise ValueError(f"Unsupported task: {task}. Should be one of ['txt2img', 'txt2txt', 'img2img', 'img2txt']")
        end_time_query = time.time()
        print(f"Finished extracting query features for {task} on {dataset_name}.".upper())
        print("=" * 50)
        print(f"Total time for extracting query features: {end_time_query - start_time_query: .2f} seconds\n Per sample time: {(end_time_query - start_time_query)/len(dataloader.dataset): .2f} seconds\n")
        print("=" * 50)
        
        test_loader = get_test_loader(cfg, dataset_name, split, extractor, img_batch_size)
        start_time_target = time.time()
        if task.endswith('2img'):
            delete_models(text_generation_model)
            if task == 'img2img':
                delete_models(image_generation_model)

            for j, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])

                target_feat.append(extract_image_features(pil, extractor, feature_extraction_model, img_preprocess))

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//img_batch_size + 1 if len(img_subset)%img_batch_size != 0 else len(img_subset)//img_batch_size
                for i in range(n_iters):
                    img_subset_feat.append(extract_image_features(img_subset[i*img_batch_size:(i+1)*img_batch_size], extractor, feature_extraction_model, img_preprocess))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")
                
            delete_models(feature_extraction_model)
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
        end_time_target = time.time()
        print("Finished extracting target image features")
        print("=" * 50)
        print(f"Total time for extracting target image features: {end_time_target - start_time_target: .2f} seconds\nPer sample time: {(end_time_target - start_time_target)/len(test_loader.dataset): .2f} seconds\n")
        print("=" * 50)

    query_feat = torch.cat(query_feat, dim=0)  
    target_feat = torch.cat(target_feat, dim=0)
    print(query_feat.shape, target_feat.shape)
    
    if dataset_name.lower() == 'circo' and split.lower() == 'test':
        store_top_k(cfg, query_ids, target_ids, query_feat, target_feat, **kwargs)
    elif dataset_name.lower() == 'cirr' and split.lower() == 'test':
        store_top_k(cfg, query_ids, target_ids, query_feat, target_feat, **kwargs)
        store_top_k(cfg, query_ids, img_subset_ids, query_feat, img_subset_feat, cutoff=3, **kwargs)
    elif dataset_name.lower() == 'fashioniq' and split.lower() == 'val':
        res = []
        for k in top_k:
            recall = fashioniq_eval(dataloader, query_feat, target_feat, fashioniq_ground_truth, target_ids, k)
            res.append(recall)
        return sum(res)/len(res)
    elif dataset_name.lower() == 'change_attribute' or dataset_name.lower() == 'change_object' or dataset_name.lower() == 'focus_attribute' or dataset_name.lower() == 'focus_object':
        res = []
        for k in [1, 2, 3]:
            recall = genecis_eval(query_feat, target_feat, target_length, k)
            res.append(recall)
        return sum(res)/len(res)
    else:
        metric = 'map' if dataset_name.lower() == 'circo' else 'recall' ######modify here!!!!!
        res = []
        for k in top_k:
            metric_val = get_metrics(query_feat,
                                      target_feat,
                                      k=k,
                                      target_length=target_length,
                                      metrics=metric
                                    )
            print("=" * 50)
            print(f'|{metric.upper()}@{k}: {metric_val:.2f}% when using generated description ---> target images |\n')
            print("=" * 50)
            
            res.append(metric_val.item())
        return sum(res)/len(res)

    print(f"{'*'*20}Completed{'*'*20}")
def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    seed = kwargs.get('seed', cfg['GENERAL']['SEED'])
    set_seed(seed)
    start = time.time()
    main(cfg, **kwargs)
    end = time.time()
    print("=" * 50)
    print(f"Total execution time: {end - start:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    fire.Fire(launch)