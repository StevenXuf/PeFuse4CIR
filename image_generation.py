import torch
import os
import torchvision.transforms.functional as F

from tqdm import tqdm
from torchvision.transforms import transforms
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from transformers import AutoModel

from figures import show_tensor_images
from configuration import get_default_config
from refinedfashioniq import get_refined_fashioniq_loader, transform_image
from feature_extraction import get_metrics

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

if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])

    image_size = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['IMAGE_SIZE']
    model_id = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['MODEL_NAME']
    store_path = cfg['IMAGE-GENERATION']['SDXL-INSTRUCTPIX2PIX']['OUTPUT_DIR']

    dataset_name = cfg['GENERAL']['DATASET']
    if dataset_name.lower() == 'circo':
        metric = 'map'
    else:
        metric = 'recall'
    top_k = cfg['GENERAL']['TOP_K']
    device = torch.device(f"cuda:{cfg['GENERAL']['DEVICE']}" if torch.cuda.is_available() else "cpu")

    img_transform_for_generation = transform_image(image_size)
    dataloader = get_refined_fashioniq_loader(cfg['RefinedFashionIQ']['OUTPUT_DIR'], transform=img_transform_for_generation, batch_size=cfg['GENERAL']['BATCH_SIZE'])

    img_transform_for_extraction = transform_image(cfg['CLIP']['IMAGE_SIZE'], cfg['CLIP']['IMAGE_MEAN'], cfg['CLIP']['IMAGE_STD'])

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    generation_model=StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    feature_extraction_model = AutoModel.from_pretrained(cfg['CLIP']['MODEL_NAME']).to(device)

    n_infer_step = cfg['IMAGE-GENERATION']['GLOBAL']['NUM_INFERENCE_STEPS']
    image_guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['IMAGE_GUIDANCE_SCALE']
    guidance_scale = cfg['IMAGE-GENERATION']['GLOBAL']['GUIDANCE_SCALE']
    print(f"Using {generation_model.__class__.__name__} with params: n_infer_step={n_infer_step}, image_guidance_scale={image_guidance_scale}, guidance_scale={guidance_scale}")

    generated_image_features = []
    target_features = []
    target_length = []
    with torch.no_grad():
        for i,batch in tqdm(enumerate(dataloader)):
            input_images=batch['reference_img']
            show_tensor_images(input_images, num_images=input_images.size(0), file_path=os.path.join(store_path,f"input_image_grid_{i}.png"))
            prompts=batch['caption']
            targets=batch['target_img']
            target_length.extend(batch['all_target_length'])

            images = generation_model(
                prompt=prompts,
                image=input_images.to(device),
                width=image_size,
                height=image_size,
                num_inference_steps=n_infer_step,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale).images

            generated_images = torch.stack(convert_pil_to_tensor(images))
            show_tensor_images(generated_images, num_images=generated_images.size(0), file_path=os.path.join(store_path,f"output_image_grid_{i}.png"))

            transformed_generated_images = torch.stack(convert_pil_to_tensor(images, transform=img_transform_for_extraction))
            generated_image_features.append(feature_extraction_model.get_image_features(pixel_values=transformed_generated_images.to(device)))

            targets = resize_crop_normalize(targets, size=cfg['CLIP']['IMAGE_SIZE'], IMAGE_MEAN=cfg['CLIP']['IMAGE_MEAN'], IMAGE_STD=cfg['CLIP']['IMAGE_STD'])
            # targets = torch.stack(convert_pil_to_tensor(targets, transform=img_transform_for_extraction))
            target_features.append(feature_extraction_model.get_image_features(pixel_values=targets.to(device)))
            if i==5:
                break
    generated_image_features = torch.cat(generated_image_features, dim=0)
    target_features = torch.cat(target_features, dim=0)

    for k in top_k:
        metric_val = get_metrics(generated_image_features, target_features, k=k, target_length=target_length, metric=metric)
        print(f'{metric}@{k}: {metric_val:.2f} when using generated images ---> real images retrieval')