import torch
import torch.nn.functional as F
import open_clip

from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalRecall, RetrievalMAP, RetrievalPrecision
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from open_clip import create_model_from_pretrained, get_tokenizer

from configuration import get_default_config
from refinedfashioniq import get_refined_fashioniq_loader
from attention import self_attention, cross_attention, co_attention
from metrics import compute_map_at_k, compute_recall_at_k
from utils import targetpad_transform, transform_image

def get_feature_extractor(cfg, extractor=None, extractor_id=None, pretrained=None):
    if extractor is None:
        extractor = cfg['GENERAL']['EXTRACTOR']
    if extractor_id is None:
        extractor_id = cfg[extractor]['MODEL_NAME']
    if extractor.lower() == 'openclip':
        if pretrained is None:
            pretrained = cfg[extractor]['PRETRAINED']
        print(f"Using pretrained model: {pretrained}")

    if extractor.lower() == 'openvision':
        print(f"Using OpenVision for feature extraction")
        feature_extraction_model, img_preprocess = create_model_from_pretrained(f'hf-hub:{extractor_id}')
        feature_extraction_model = feature_extraction_model
        tokenizer = get_tokenizer(f'hf-hub:{extractor_id}')
    elif extractor.lower() == 'openclip':
        print(f"Using OpenCLIP for feature extraction")
        feature_extraction_model, _, img_preprocess = open_clip.create_model_and_transforms(extractor_id, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(extractor_id)
    elif extractor.lower() == 'clip':
        print(f"Using CLIP for feature extraction")
        feature_extraction_model = AutoModel.from_pretrained(extractor_id)
        tokenizer = AutoTokenizer.from_pretrained(extractor_id)
        img_preprocess = targetpad_transform(cfg[extractor]['IMAGE_MEAN'], cfg[extractor]['IMAGE_STD'], target_ratio=1.25, dim=cfg[extractor]['IMAGE_SIZE'])
    elif extractor.lower() == 'siglip2':
        print(f"Using SigLip2 for feature extraction")
        feature_extraction_model = AutoModel.from_pretrained(extractor_id)
        tokenizer = AutoTokenizer.from_pretrained(extractor_id)
        img_preprocess = AutoProcessor.from_pretrained(extractor_id)
    return feature_extraction_model, img_preprocess, tokenizer

def get_metrics(feat1, feat2, k, target_length, metrics='recall'):
    if metrics == 'recall':
        compute = RetrievalRecall(top_k=k)
    elif metrics == 'precision':
        compute = RetrievalPrecision(top_k=k)
    elif metrics == 'map':
        compute = RetrievalMAP(top_k=k)

    sim=pairwise_cosine_similarity(feat1, feat2)

    targets = torch.zeros(sim.size(0), sim.size(1), dtype=torch.long).to(sim.device)
    start = 0
    for i, length in enumerate(target_length):
        targets[i, start:start+length] = 1
        start += length

    indexes = torch.arange(sim.size(0), dtype=torch.long).unsqueeze(1).expand(*sim.size()).to(sim.device)

    res = compute(sim,targets,indexes=indexes)

    # if metrics == 'map':
    #     print(f'manual mAP@{k}: {compute_map_at_k(sim, targets, k=k)*100}')
    # elif metrics == 'recall':
    #     print(f'manual Recall@{k}: {compute_recall_at_k(sim, targets, k=k)*100}')
    return res*100

def extract_features(model,processor,dataloader,config_path='./config.yaml',cfg=None):

    if cfg is None:
        cfg=get_default_config(config_path)
    
    device=torch.device(f'cuda:{cfg["GENERAL"]["DEVICE"]}' if torch.cuda.is_available() else 'cpu')

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
    dataloader = get_refined_fashioniq_loader(cfg['RefinedFashionIQ']['OUTPUT_DIR'], transform=img_transform, batch_size=cfg['GENERAL']['BATCH_SIZE'])
    ref_image_features, candidate_image_features, text_features = extract_features(model, processor, dataloader=dataloader, cfg=cfg)
    perform_retrieval(ref_image_features, candidate_image_features, text_features, cfg['GENERAL']['TOP_K'])

if __name__=='__main__':
    # cfg= get_default_config("config.yaml")
    # torch.manual_seed(cfg['GENERAL']['SEED'])
    # main(cfg)
    torch.manual_seed(42)
    feat1 = torch.randn(5, 512)
    feat2 = torch.randn(10, 512)
    length = [1,2,2,3,2]
    res = get_metrics(feat1, feat2, 5, length)
    print(res)