import torch
from tqdm import tqdm

import torch.nn.functional as F

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalRecall
from transformers import AutoProcessor, AutoModel

from configuration import get_default_config
from fashioniq import get_fashioniq_loader,transform_image
from attention import self_attention_batched, cross_attention_batched, co_attention_batched

def get_metrics(text_features,audio_features,k):     
    compute_recall=RetrievalRecall(top_k=k)
    sim=pairwise_cosine_similarity(text_features,audio_features)

    targets=torch.diag(torch.ones(sim.size(0), dtype=torch.long)).to(sim.device)

    indexes = torch.arange(sim.size(0), dtype=torch.long).unsqueeze(1).expand(*sim.size()).to(sim.device)

    recall=compute_recall(sim.flatten(),targets.flatten(),indexes=indexes.flatten())

    return recall



def extract_features(model,processor,config_path='./config.yaml',TOP_K=None):

    cfg=get_default_config(config_path)
    if TOP_K is None:
        TOP_K = cfg['General']['TOP_K']
    device=torch.device(f'cuda:{cfg["General"]["DEVICE"]}' if torch.cuda.is_available() else 'cpu')
    mean = cfg['CLIP']['IMAGE_MEAN']
    std = cfg['CLIP']['IMAGE_STD']
    img_transform = transform_image(cfg['CLIP']['IMAGE_SIZE'], mean, std)
    dataloader = get_fashioniq_loader(cfg['FashionIQ']['OUTPUT_DIR'], transform=img_transform, batch_size=cfg['General']['BATCH_SIZE'])

    model.to(device)

    ref_image_features = []
    candidate_image_features = []
    text_features = []
    with torch.no_grad():
        print(f"Processing batches...")
        for batch in tqdm(dataloader):
            target_images = batch['target'].to(device)
            reference_images = batch['reference'].to(device)
            captions = batch['caption']

            text_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)

            text_feat = model.get_text_features(**text_inputs)
            ref_feat = model.get_image_features(pixel_values=reference_images)
            candidate_feat = model.get_image_features(pixel_values=target_images)

            ref_image_features.append(ref_feat)
            text_features.append(text_feat)
            candidate_image_features.append(candidate_feat)
        print(f"Finished processing batches.")

    ref_image_features = torch.cat(ref_image_features, dim=0)
    candidate_image_features = torch.cat(candidate_image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    recall_ref2can=get_metrics(ref_image_features,candidate_image_features,TOP_K)
    recall_t2ref=get_metrics(text_features,candidate_image_features,TOP_K)

    print(f'Recall@{TOP_K}: image-to-image {recall_ref2can:.2f}: text-to-image {recall_t2ref:.2f}')

    #image+text---->image
    print(f'Using image+text pairs to retrieve image')
    added_features=F.normalize(ref_image_features+text_features,p=2,dim=1)
    recall_added=get_metrics(added_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_added:.2f} when using add(image,text)---->image')

    #image*text---->image
    print(f'Using image*text pairs to retrieve image')
    multiplied_features=F.normalize(ref_image_features*text_features,p=2,dim=1)
    recall_multiplied=get_metrics(multiplied_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_multiplied:.2f} when using multiply(image,text)---->image')

    #attention-based retrieval
    # print(f'Using attention-based retrieval')
    # attention_features = F.normalize(self_attention(image_features, text_features), p=2, dim=1)
    # recall_attention = get_metrics(attention_features, audio_features, TOP_K)
    
if __name__=='__main__':
    torch.manual_seed(0)

    MODEL_FILENAME = "openai/clip-vit-base-patch32"
    model = AutoModel.from_pretrained(MODEL_FILENAME)
    processor = AutoProcessor.from_pretrained(MODEL_FILENAME)
    extract_features(model, processor)
