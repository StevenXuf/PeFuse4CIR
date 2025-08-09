import torch
from tqdm import tqdm

import torch.nn.functional as F

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalRecall
from transformers import AutoProcessor, AutoModel

from configuration import get_default_config
from fashioniq import get_fashioniq_loader,transform_image
from attention import self_attention, cross_attention, co_attention

def get_metrics(text_features,audio_features,k):     
    compute_recall=RetrievalRecall(top_k=k)
    sim=pairwise_cosine_similarity(text_features,audio_features)

    targets=torch.diag(torch.ones(sim.size(0), dtype=torch.long)).to(sim.device)

    indexes = torch.arange(sim.size(0), dtype=torch.long).unsqueeze(1).expand(*sim.size()).to(sim.device)

    recall=compute_recall(sim.flatten(),targets.flatten(),indexes=indexes.flatten())

    return recall*100



def extract_features(model,processor,dataloader,config_path='./config.yaml',cfg=None):

    if cfg is None:
        cfg=get_default_config(config_path)
    
    device=torch.device(f'cuda:{cfg["General"]["DEVICE"]}' if torch.cuda.is_available() else 'cpu')

    print(f'Using {cfg["CLIP"]["MODEL_NAME"].upper()} for feature extraction')
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

    return ref_image_features, candidate_image_features, text_features

def perform_retrieval(ref_image_features, candidate_image_features, text_features, TOP_K):

    ##################feature based retrival####################
    print('\n' + '='*50 + '\nFeature-based Image Retrieval\n' + '='*50)
    recall_ref2can=get_metrics(ref_image_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_ref2can:.2f} when using image-to-image retrieval')
    recall_t2ref=get_metrics(text_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_t2ref:.2f} when using text-to-image retrieval')

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

    ########################self-attention retrieval###########################
    print('\n' + '='*50 + '\nSelf-Attention Retrieval\n' + '='*50)
    print(f'Using self-attention on ref images')
    self_attn_candidate_features, _ = self_attention(candidate_image_features)
    self_attn_img_features, weights = self_attention(ref_image_features)
    recall_img_attention = get_metrics(self_attn_img_features, self_attn_candidate_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_img_attention:.2f} when using self-attention(ref_image)---->image')

    print(f'Using self-attention on text features')
    self_attn_text_features, weights = self_attention(text_features)
    recall_text_attention = get_metrics(self_attn_text_features, self_attn_candidate_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_text_attention:.2f} when using self-attention(text)---->image')

    print(f'Using addition between self-attended features')
    self_added_features = F.normalize(self_attn_img_features + self_attn_text_features, p=2, dim=1)
    recall_added = get_metrics(self_added_features, self_attn_candidate_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_added:.2f} when using add(self_attn(ref_image),self_attn(text))---->image')

    print(f'Using multiplication between self-attended features')
    self_multiplied_features = F.normalize(self_attn_img_features * self_attn_text_features, p=2, dim=1)
    recall_multiplied = get_metrics(self_multiplied_features, self_attn_candidate_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_multiplied:.2f} when using multiply(self_attn(ref_image),self_attn(text))---->image')

    #####################cross-attention retrieval########################
    print('\n' + '='*50 + '\nCross-Attention based on ref_image_features and text_features\n' + '='*50)
    print(f'Using cross-attention between text and ref images')
    text_attn_image_features, img_attn_txt_features, text_weights, img_weights = co_attention(ref_image_features, text_features)
    recall_cross_attention = get_metrics(text_attn_image_features, candidate_image_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_cross_attention:.2f} when using cross_attn(text,ref_image)---->image')
    recall_img_cross_attention = get_metrics(img_attn_txt_features, candidate_image_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_img_cross_attention:.2f} when using cross_attn(ref_image,text)---->image')

    print(f'Using addition between cross-attended features')
    added_features = F.normalize(img_attn_txt_features + text_attn_image_features, p=2, dim=1)
    recall_added = get_metrics(added_features, candidate_image_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_added:.2f} when using add(cross_attn(ref_image,text),cross_attn(text,ref_image))---->image')

    print(f'Using multiplication between cross-attended features')
    multiplied_features = F.normalize(img_attn_txt_features * text_attn_image_features, p=2, dim=1)
    recall_multiplied = get_metrics(multiplied_features, candidate_image_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_multiplied:.2f} when using multiply(cross_attn(ref_image,text),cross_attn(text,ref_image))---->image')

    ##################cross attention using ref_image_features as context####################
    print('\n' + '='*50 + '\nCross-Attention using candidate_image_features as context)\n' + '='*50)
    ref_img_attn_can_features, _ = cross_attention(ref_image_features, candidate_image_features)
    recall_ref_img_cross_attention = get_metrics(ref_img_attn_can_features, candidate_image_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_ref_img_cross_attention:.2f} when using cross_attn(ref_image,can_image)---->image')

    txt_attn_can_features, _ = cross_attention(text_features, candidate_image_features)
    recall_txt_cross_attention = get_metrics(txt_attn_can_features, candidate_image_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_txt_cross_attention:.2f} when using cross_attn(text,can_image)---->image')

    recall_ref_and_txt_cross_attn_combined = get_metrics(ref_img_attn_can_features + txt_attn_can_features, candidate_image_features, TOP_K)
    print(f'Recall@{TOP_K}: {recall_ref_and_txt_cross_attn_combined:.2f} when using add(cross_attn(ref_image,can_image),cross_attn(text,can_image))---->image')

def main(cfg):
    MODEL_FILENAME = cfg['CLIP']['MODEL_NAME']
    model = AutoModel.from_pretrained(MODEL_FILENAME)
    processor = AutoProcessor.from_pretrained(MODEL_FILENAME)
    mean = cfg['CLIP']['IMAGE_MEAN']
    std = cfg['CLIP']['IMAGE_STD']
    img_transform = transform_image(cfg['CLIP']['IMAGE_SIZE'], mean, std)
    dataloader = get_fashioniq_loader(cfg['FashionIQ']['OUTPUT_DIR'], transform=img_transform, batch_size=cfg['General']['BATCH_SIZE'])
    ref_image_features, candidate_image_features, text_features = extract_features(model, processor, dataloader=dataloader, cfg=cfg)
    perform_retrieval(ref_image_features, candidate_image_features, text_features, cfg['General']['TOP_K'])

if __name__=='__main__':
    cfg= get_default_config("config.yaml")
    torch.manual_seed(cfg['General']['SEED'])
    main(cfg)

