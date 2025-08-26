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
from refinedfashioniq import transform_image
from feature_extraction import get_metrics, get_feature_extractor
from dataloaders import get_dataloader
from text_generation_val import fashioniq_eval
from prompts import get_composed_prompts
from image_generation_val import convert_pil_to_tensor, resize_crop_normalize

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
    
    if kwargs.get('DEVICE'):
        device = kwargs['DEVICE']
    else:
        device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
    if kwargs.get('EXTRACTOR'):
        extractor_name = kwargs['EXTRACTOR']
    else:
        extractor_name = cfg['GENERAL']['EXTRACTOR']
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

    img_transform_for_generation = transform_image(image_size)
    dataloader = get_dataloader(cfg, 
                                split=split,
                                batch_size=batch_size,
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
                                                                               ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, 
                                              padding_side='left', 
                                              use_fast=True
                                              )

    if extractor_name.lower() == 'openvision' or extractor_name.lower() == 'openclip':
        feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, extractor=extractor_name)
    else:
        feature_extraction_model, tokenizer = get_feature_extractor(cfg, extractor=extractor_name)
    feature_extraction_model.eval()
    feature_extraction_model.to(device)

    generated_captions= []
    with torch.no_grad(), torch.autocast("cuda"):
        for i,batch in tqdm(enumerate(dataloader)):
            reference_pil = batch['reference_pil']
            caption = batch['caption']

            composed_description = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in composed_description
            ]
            image_inputs, video_inputs = process_vision_info(composed_description)
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
            generated_captions.extend(output_texts)
            print(output_texts)

    del text_generation_model
    gc.collect()
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cuda").manual_seed(cfg['GENERAL']['SEED'])
    generation_model=StableDiffusionXLInstructPix2PixPipeline.from_pretrained(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['MODEL_NAME'], 
                                                                              torch_dtype=torch.float16).to(device)
    print(f"Using {generation_model.__class__.__name__} with params: n_infer_step={n_infer_step}, image_guidance_scale={image_guidance_scale}, guidance_scale={guidance_scale}")

    store_path = os.path.join(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['OUTPUT_DIR'], f'Qwen_{dataset_name}_{extractor_name}_{n_infer_step}_{image_guidance_scale}_{guidance_scale}')
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    generated_target_features = []
    target_features = []
    target_length = []
    start = 0
    with torch.no_grad(), torch.autocast("cuda"):
        for j,batch in tqdm(enumerate(dataloader)):
            input_images = batch['reference_img']
            target_prompts = generated_captions[start:start+input_images.size(0)]
            targets = batch['target_img']
            all_target_pil = batch['all_target_pil']
            all_target_img = batch['all_target_img']
            target_length.extend(batch['all_target_length'])

            show_tensor_images(input_images, num_images=input_images.size(0), file_path=os.path.join(store_path,f"reference_image_grid_{j}.png"))
            show_tensor_images(targets, num_images=targets.size(0), file_path=os.path.join(store_path,f"target_image_grid_{j}.png"))

            print(f"Generating target images for batch {j+1}")
            generated_target_images = generation_model(
                prompt=target_prompts,
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

            if extractor_name.lower() == 'openvision' or extractor_name.lower() == 'openclip':
                generated_target_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in generated_target_images],dim=0).to(device)))
                target_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in all_target_pil],dim=0).to(device))) #multiple targets for circo

            else:
                transformed_generated_images = torch.stack(convert_pil_to_tensor(generated_target_images, transform=img_transform_for_extraction))
                all_target_img = resize_crop_normalize(all_target_img, 
                                                       size=cfg[extractor_name]['IMAGE_SIZE'], 
                                                       IMAGE_MEAN=cfg[extractor_name]['IMAGE_MEAN'], 
                                                       IMAGE_STD=cfg[extractor_name]['IMAGE_STD']
                                                       )
                generated_target_features.append(feature_extraction_model.get_image_features(pixel_values=transformed_generated_images.to(device)))
                target_features.append(feature_extraction_model.get_image_features(pixel_values=all_target_img.to(device)))

            start += input_images.size(0)

            print(f'Batch {j+1} finished.')

    print(target_length)
    generated_target_features = torch.cat(generated_target_features, dim=0)
    target_features = torch.cat(target_features, dim=0)

    for k in top_k:
        if dataset_name.lower() == "fashioniq" and split == "val":
            fashioniq_eval(dataloader, generated_target_features, target_features, target_length, k)
        else:
            tar_metric_val = get_metrics(generated_target_features, target_features, k=k, target_length=target_length, metrics='map' if dataset_name.lower() == "circo" else 'recall')
            print(f'{"mAP" if dataset_name.lower() == "circo" else "Recall"}@{k}: {tar_metric_val:.2f}% when using generated images ---> real images retrieval')


def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    set_seed(cfg['GENERAL']['SEED'])
    main(cfg, **kwargs)

if __name__ == "__main__":
    fire.Fire(launch)