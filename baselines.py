import fire
import torch
import time
from tqdm import tqdm

from torch.nn import functional as F
from transformers import set_seed

from dataloaders import get_dataloader
from utils import get_default_config, transform_image
from feature_extraction import get_feature_extractor, get_metrics
from text_to_image_and_text import extract_text_features, extract_image_features

@torch.inference_mode()
def compute_baseline(**kwargs):
    cfg = get_default_config("config.yaml")
    split = kwargs.get('split', cfg['GENERAL']['SPLIT'])
    dataset_name = kwargs.get('dataset', cfg['GENERAL']['DATASET'])
    extractor = kwargs.get('extractor', cfg['GENERAL']['EXTRACTOR'])
    batch_size = 256
    seed = kwargs.get('seed', cfg['GENERAL']['SEED'])
    extractor_id = kwargs.get('extractor_id', cfg[extractor]['MODEL_NAME'])
    pretrained = kwargs.get('pretrained', cfg[extractor]['PRETRAINED'])
    device = torch.device(f"cuda:{kwargs.get('device')}") if kwargs.get('device') is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, 
                                                                                extractor=extractor, 
                                                                                extractor_id=extractor_id, 
                                                                                pretrained=pretrained
                                                                                )
    feature_extraction_model.to(device)
    feature_extraction_model.eval()

    query_feat = {"modification": [], "reference": []}
    gt_img_ids = []
    query_loader = get_dataloader(cfg, 
                                split='val', 
                                mode='relative',
                                dataset_name=dataset_name, 
                                extractor_name=extractor,
                                batch_size=batch_size,
                                transform=img_preprocess
                                )
    for query_batch in tqdm(query_loader, desc="Extracting query features"):
        gt_img_ids.extend(query_batch['target_id'])
        query_feat["modification"].append(extract_text_features(query_batch['relative_caption'], extractor, tokenizer, feature_extraction_model))
        query_feat["reference"].append(extract_image_features(query_batch['reference_pil'], extractor, feature_extraction_model, img_preprocess))
    query_feat["modification"] = torch.cat(query_feat["modification"], dim=0)
    query_feat["reference"] = torch.cat(query_feat["reference"], dim=0)

    target_loader = get_dataloader(cfg, 
                                dataset_name=dataset_name, 
                                split='val', 
                                mode='classic',
                                batch_size=batch_size, 
                                extractor_name=extractor,
                                transform=img_preprocess
                                )
    target_ids = []
    target_length = []
    target_feat = []
    for target_batch in tqdm(target_loader, desc="Extracting target features"):
        target_ids.extend(target_batch['target_id'])
        target_length.extend(target_batch['target_length'])
        target_feat.append(extract_image_features(target_batch['target_pil'], extractor, feature_extraction_model, img_preprocess))
    target_feat = torch.cat(target_feat, dim=0)
    print(target_feat.shape)

    id_to_index = {img_id: idx for idx, img_id in enumerate(target_ids)}
    index= [[id_to_index[i.zfill(12)] for i in group] for group in gt_img_ids]

    all_query_feats = [F.normalize(query_feat["modification"] + query_feat["reference"], p=2, dim=1)]
    for q_feat in all_query_feats:
        res = []
        for k in [1, 5, 10, 25, 50]:
            metric_val, _ = get_metrics(q_feat,
                                        target_feat,
                                        k=k,
                                        target_length=target_length,
                                        metrics='map',
                                        gt_img_ids=index
                                    )
            print(f'mAP@{k}: {metric_val:.2f}%.')
            res.append(metric_val.item())
        print(f"mAP AVG: {torch.tensor(res).mean().item():.2f}%.")
        print("=" * 50)

if __name__ == "__main__":
    t = []
    for i in range(3):
        start_time = time.time()
        fire.Fire(compute_baseline)
        end_time = time.time()
        t.append((end_time - start_time) / 220)
    print(f"Time: {torch.tensor(t).mean():.2f} with std {torch.tensor(t).std():.2f} seconds.")