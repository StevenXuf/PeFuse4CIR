import torch
import os
import torchvision.transforms.functional as F

from tqdm import tqdm
from torchvision.transforms import transforms
from diffusers import StableDiffusionXLInstructPix2PixPipeline

from figures import show_tensor_images
from configuration import get_default_config
from refinedfashioniq import transform_image
from feature_extraction import get_metrics, get_feature_extractor
from dataloaders import get_dataloader

def convert_pil_to_tensor(list_of_pils, transform=None):
    if transform is not None:
        return [transform(image) for image in list_of_pils]
    else:
        return [transforms.ToTensor()(image) for image in list_of_pils]
    
def resize_crop_normalize(tensor_img, size=224, IMAGE_MEAN=None, IMAGE_STD=None):
    # tensor_img: C×H×W in [0,1]
    img = F.resize(tensor_img, [size, size], interpolation=transforms.InterpolationMode.BICUBIC)
    img = F.center_crop(img, size)
    img = F.normalize(img, mean=IMAGE_MEAN, std=IMAGE_STD)
    return img

def main(cfg):
    image_size = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['IMAGE_SIZE']
    model_id = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['MODEL_NAME']
    top_k = cfg['GENERAL']['TOP_K']
    device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")
    extractor_name = cfg['GENERAL']['EXTRACTOR']
    extractor_id = cfg[extractor_name]['MODEL_NAME']
    dataset_name = cfg['GENERAL']['DATASET']
    print(f"Using {extractor_name} with id: {extractor_id} for feature extraction on {dataset_name} dataset.")

    store_path = os.path.join(cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['OUTPUT_DIR'], dataset_name)
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    n_infer_step = cfg['IMAGE-GENERATION']['GLOBAL']['NUM_INFERENCE_STEPS']
    image_guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['IMAGE_GUIDANCE_SCALE']
    guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['GUIDANCE_SCALE']
    generation_model=StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    print(f"Using {generation_model.__class__.__name__} with params: n_infer_step={n_infer_step}, image_guidance_scale={image_guidance_scale}, guidance_scale={guidance_scale}")

    img_transform_for_generation = transform_image(image_size)
    dataloader = get_dataloader(cfg, transform=img_transform_for_generation)

    img_transform_for_extraction = transform_image(cfg[extractor_name]['IMAGE_SIZE'], 
                                                   cfg[extractor_name]['IMAGE_MEAN'], 
                                                   cfg[extractor_name]['IMAGE_STD']
                                                   )

    if extractor_name.lower() == 'openvision' or extractor_name.lower() == 'openclip':
        feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg)
    else:
        feature_extraction_model, tokenizer = get_feature_extractor(cfg)
    feature_extraction_model.eval()
    feature_extraction_model.to(device)

    generated_target_features = []
    # generated_reference_features = []
    target_features = []
    # reference_features = []
    target_length = []
    with torch.no_grad(), torch.autocast("cuda"):
        for i,batch in tqdm(enumerate(dataloader)):
            input_images = batch['reference_img']
            reference_pil = batch['reference_pil']
            target_prompts = batch['caption']
            targets = batch['target_img']
            all_target_pil = batch['all_target_pil']
            all_target_img = batch['all_target_img']
            target_length.extend(batch['all_target_length'])

            show_tensor_images(input_images, num_images=input_images.size(0), file_path=os.path.join(store_path,f"reference_image_grid_{i}.png"))
            show_tensor_images(targets, num_images=targets.size(0), file_path=os.path.join(store_path,f"target_image_grid_{i}.png"))

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

            if extractor_name.lower() == 'openvision' or extractor_name.lower() == 'openclip':
                generated_target_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in generated_target_images],dim=0).to(device)))
                target_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in all_target_pil],dim=0).to(device))) #multiple targets for circo

                # generated_reference_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in generated_reference_images],dim=0).to(device)))
                # reference_features.append(feature_extraction_model.encode_image(torch.cat([img_preprocess(img).unsqueeze(0) for img in reference_pil],dim=0).to(device)))
            else:
                transformed_generated_images = torch.stack(convert_pil_to_tensor(generated_target_images, transform=img_transform_for_extraction))
                all_target_img = resize_crop_normalize(all_target_img, 
                                                       size=cfg[extractor_name]['IMAGE_SIZE'], 
                                                       IMAGE_MEAN=cfg[extractor_name]['IMAGE_MEAN'], 
                                                       IMAGE_STD=cfg[extractor_name]['IMAGE_STD']
                                                       )
                # using convert_pil_to_tensor works not as good as using resize_crop_normalize
                # targets = torch.stack(convert_pil_to_tensor(targets, transform=img_transform_for_extraction))
                generated_target_features.append(feature_extraction_model.get_image_features(pixel_values=transformed_generated_images.to(device)))
                target_features.append(feature_extraction_model.get_image_features(pixel_values=all_target_img.to(device)))

                # transformed_generated_reference_images = torch.stack(convert_pil_to_tensor(generated_reference_images, transform=img_transform_for_extraction))
                # generated_reference_features.append(feature_extraction_model.get_image_features(pixel_values=transformed_generated_reference_images.to(device)))
                # input_images = resize_crop_normalize(input_images, 
                #                                       size=cfg[extractor_name]['IMAGE_SIZE'], 
                #                                       IMAGE_MEAN=cfg[extractor_name]['IMAGE_MEAN'], 
                #                                       IMAGE_STD=cfg[extractor_name]['IMAGE_STD']
                #                                       )
                # reference_features.append(feature_extraction_model.get_image_features(pixel_values=input_images.to(device)))

            print(f'Batch {i+1} finished.')
            # if i==5:
            #     break

    print(target_length)
    generated_target_features = torch.cat(generated_target_features, dim=0)
    target_features = torch.cat(target_features, dim=0)
    # generated_reference_features = torch.cat(generated_reference_features, dim=0)
    # reference_features = torch.cat(reference_features, dim=0)

    for k in top_k:
        tar_metric_val = get_metrics(generated_target_features, target_features, k=k, target_length=target_length, metrics='map' if dataset_name.lower() == "circo" else 'recall')
        print(f'{"mAP" if dataset_name.lower() == "circo" else "Recall"}@{k}: {tar_metric_val:.2f}% when using generated images ---> real images retrieval')

        # ref_metric_val = get_metrics(generated_reference_features, reference_features, k=k, target_length=[1]*reference_features.size(0), metrics='recall')
        # print(f'RECALL@{k}: {ref_metric_val:.2f}% when using generated reference images ---> real reference retrieval')


if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    main(cfg)