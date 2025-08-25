import torch
import os
import fire
import json
import gc
import numpy as np

from tqdm import tqdm
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from figures import show_tensor_images
from configuration import get_default_config
from refinedfashioniq import transform_image
from feature_extraction import get_feature_extractor
from dataloaders import get_dataloader
from image_generation_val import convert_pil_to_tensor
    
def main(cfg, **kwargs):
    image_size = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['IMAGE_SIZE']
    model_id = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['MODEL_NAME']
    device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")

    if kwargs.get('EXTRACTOR'):
        extractor = kwargs['EXTRACTOR']
    else:
        extractor = cfg['GENERAL']['EXTRACTOR']
    if kwargs.get('DATASET'):
        dataset_name = kwargs['DATASET']
    else:
        dataset_name = cfg['GENERAL']['DATASET']
    extractor_id = cfg[extractor]['MODEL_NAME']
    print(f"Using {extractor} with id: {extractor_id} for feature extraction on {dataset_name} dataset.")

    if kwargs.get('BATCH_SIZE'):
        batch_size = kwargs['BATCH_SIZE']
    else:
        batch_size = cfg['GENERAL']['BATCH_SIZE']
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

    generation_model=StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    print(f"Using {generation_model.__class__.__name__} with params: n_infer_step={n_infer_step}, image_guidance_scale={image_guidance_scale}, guidance_scale={guidance_scale}")

    store_path = os.path.join(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['OUTPUT_DIR'], f'{dataset_name}_{n_infer_step}_{image_guidance_scale}_{guidance_scale}')
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    img_transform_for_generation = transform_image(image_size)
    split = 'test1' if dataset_name.lower() == 'cirr' else 'test'
    dataloader = get_dataloader(cfg, 
                                split=split, 
                                dataset_name=dataset_name, 
                                extractor_name=extractor, 
                                transform=img_transform_for_generation,
                                batch_size=batch_size)

    img_transform_for_extraction = transform_image(cfg[extractor]['IMAGE_SIZE'], 
                                                   cfg[extractor]['IMAGE_MEAN'], 
                                                   cfg[extractor]['IMAGE_STD']
                                                   )

    if extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
        feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, extractor=extractor)
    else:
        feature_extraction_model, tokenizer = get_feature_extractor(cfg, extractor=extractor)
    feature_extraction_model.eval()
    feature_extraction_model.to(device)

    generated_target_features = []
    # generated_reference_features = []
    # reference_features = []
    query_ids = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i,batch in tqdm(enumerate(dataloader)):
            input_images = batch['reference_img']
            target_prompts = batch['caption']
            query_ids.extend(batch['query_id'])

            show_tensor_images(input_images, num_images=input_images.size(0), file_path=os.path.join(store_path,f"test_reference_image_grid_{i}.png"))

            print(f"Generating target images for batch {i+1}")
            generated_target_images = generation_model(
                prompt=target_prompts,
                image=input_images.to(device),
                width=image_size,
                height=image_size,
                num_inference_steps=n_infer_step,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale).images
            generated_target_image_tensor = torch.stack(convert_pil_to_tensor(generated_target_images))
            show_tensor_images(generated_target_image_tensor, num_images=generated_target_image_tensor.size(0), file_path=os.path.join(store_path,f"generated_target_image_grid_{i}.png"))

            ####modify the captions!
            # reference_prompts = ['Remove the following modifications: ' + caption for caption in batch['caption']]
            # print(f"Generating reference images for batch {i+1}")
            # generated_reference_images = generation_model(
            #     prompt=reference_prompts,
            #     image=targets.to(device),
            #     width=image_size,
            #     height=image_size,
            #     num_inference_steps=n_infer_step,
            #     image_guidance_scale=image_guidance_scale,
            #     guidance_scale=guidance_scale).images
            # generated_reference_image_tensor = torch.stack(convert_pil_to_tensor(generated_reference_images))
            # show_tensor_images(generated_reference_image_tensor, num_images=generated_reference_image_tensor.size(0), file_path=os.path.join(store_path,f"generated_reference_image_grid_{i}.png"))

            if extractor.lower() == 'openvision' or extractor.lower() == 'openclip':
                generated_target_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in generated_target_images],dim=0).to(device)))

                # generated_reference_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in generated_reference_images],dim=0).to(device)))
                # reference_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in reference_pil],dim=0).to(device)))
            else:
                transformed_generated_images = torch.stack(convert_pil_to_tensor(generated_target_images, transform=img_transform_for_extraction))
                # using convert_pil_to_tensor works not as good as using resize_crop_normalize
                # targets = torch.stack(convert_pil_to_tensor(targets, transform=img_transform_for_extraction))
                generated_target_features.append(feature_extraction_model.get_image_features(pixel_values=transformed_generated_images.to(device)))

                # transformed_generated_reference_images = torch.stack(convert_pil_to_tensor(generated_reference_images, transform=img_transform_for_extraction))
                # generated_reference_features.append(feature_extraction_model.get_image_features(pixel_values=transformed_generated_reference_images.to(device)))
                # input_images = resize_crop_normalize(input_images, 
                #                                       size=cfg[extractor]['IMAGE_SIZE'], 
                #                                       IMAGE_MEAN=cfg[extractor]['IMAGE_MEAN'], 
                #                                       IMAGE_STD=cfg[extractor]['IMAGE_STD']
                #                                       )
                # reference_features.append(feature_extraction_model.get_image_features(pixel_values=input_images.to(device)))

            print(f'Batch {i+1} finished.')
            # if i==0:
            #     break

    #clean the generation model after use
    del generation_model
    gc.collect()
    torch.cuda.empty_cache()

    generated_target_features = torch.cat(generated_target_features, dim=0)

    if dataset_name.lower() == 'circo':
        test_loader = get_dataloader(cfg, dataset_name='circo_target_image', batch_size=1024, extractor_name=extractor)
    elif dataset_name.lower() == 'cirr':
        test_loader = get_dataloader(cfg, dataset_name='cirr_target_image', batch_size=1024, extractor_name=extractor)

    tar_tensor_feat = []
    target_ids = []
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

    target_features = torch.cat(tar_tensor_feat, dim=0)
    # generated_reference_features = torch.cat(generated_reference_features, dim=0)
    # reference_features = torch.cat(reference_features, dim=0)

    target_ids = np.array(target_ids)
    sim = pairwise_cosine_similarity(generated_target_features, target_features)
    cutoff = 50
    _, indices = sim.topk(k=cutoff, dim=1)
    predicted = [target_ids[row] for row in indices.tolist()]
    if dataset_name.lower() == 'circo':
        res={item[0]: item[1].astype(int).tolist() for item in zip(query_ids, predicted)}
    elif dataset_name.lower() == 'cirr':
        res={str(item[0]): item[1].tolist() for item in zip(query_ids, predicted)}
        res['version'] = 'rc2'
        res['metric'] = 'recall' if cutoff == 50 else 'recall_subset'
    json.dump(res, open(f"predicted_results_img_gen_{dataset_name}_{extractor}.json", "w"))

def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    main(cfg, **kwargs)

if __name__ == "__main__":
    fire.Fire(launch)