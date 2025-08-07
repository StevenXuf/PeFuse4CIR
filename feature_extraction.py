import torch
from tqdm import tqdm
import os
import sys

import torch.nn.functional as F

from torchmetrics.functional.pairwise import pairwise_cosine_similarity,pairwise_euclidean_distance
from torchmetrics.retrieval import RetrievalRecall

from get_datasets import get_all_loaders
from configuration import get_default_config
from deam import get_deam_loader
from attention import self_attention_batched, cross_attention_batched, co_attention_batched

sys.path.append(os.path.abspath("AudioCLIP"))
from model import AudioCLIP

def get_metrics(text_features,audio_features,k,metric='cosine'):     
    compute_recall=RetrievalRecall(top_k=k)
    sim=pairwise_cosine_similarity(text_features,audio_features)

    targets=torch.diag(torch.ones(sim.size(0), dtype=torch.long)).to(sim.device)

    indexes = torch.arange(sim.size(0), dtype=torch.long).unsqueeze(1).expand(*sim.size()).to(sim.device)

    recall=compute_recall(sim.flatten(),targets.flatten(),indexes=indexes.flatten())

    return recall


def get_metrics_(cosine_similarities, euclidean_distances, k):
    """
    Compute Recall@k based on Euclidean distances, using Cosine similarities to define relevance.
    Args:
        euclidean_distances: [N_queries, N_candidates]
        cosine_similarities: [N_queries, N_candidates]
        k: int, top-k value for Recall@k
    Returns:
        recall@k (float)
    """
    compute_recall = RetrievalRecall(top_k=k)

    # Define relevance using top-k cosine similarity
    #the smaller euclidean_distances, the more relevant the item is
    _, ids = torch.topk(euclidean_distances, 1, largest=False, dim=1)
    targets = torch.zeros_like(cosine_similarities, dtype=torch.long).to(cosine_similarities.device)
    targets.scatter_(1, ids.to(targets.device), 1)  # mark relevant items

    # Build group indexes (same query index for each row)
    indexes = torch.arange(cosine_similarities.size(0), dtype=torch.long, device=cosine_similarities.device).unsqueeze(1).expand_as(cosine_similarities)

    # Flatten for metric
    recall = compute_recall(
        preds=cosine_similarities.flatten(),
        target=targets.flatten(),
        indexes=indexes.flatten()
    )

    return recall

    
def extract_features(model,config_path='./config.yaml',SEQ_LENGTH=None,TOP_K=None):

    cfg=get_default_config(config_path)
    device=torch.device(f'cuda:{cfg["General"]["DEVICE"]}' if torch.cuda.is_available() else 'cpu')
    if SEQ_LENGTH is None:
        SEQ_LENGTH=cfg['AudioCLIP']['SEQ_LENGTH']
    if TOP_K is None:
        TOP_K=cfg['General']['TOP_K']

    info = get_all_loaders()
    wikiart_loader, deam_train_loader, deam_test_loader = info['wikiart'], info['deam_train'], info['deam_test']
    deam_train_loader = get_deam_loader(config_path=config_path)
    
    model.to(device)

    with torch.no_grad():
        audio_features=[]
        deam_train_va=[]
        for train_batch in tqdm(deam_train_loader):
            audio=train_batch['audio'].unsqueeze(1).to(device)
            ((audio_embeddings, _, _), _), _ = model(audio=audio)
            audio_features.append(audio_embeddings)
            deam_train_va.append(torch.cat([train_batch['valence_mean'].unsqueeze(1),train_batch['arousal_mean'].unsqueeze(1)],dim=1))
        deam_train_va=torch.cat(deam_train_va,dim=0)
        audio_features=F.normalize(torch.cat(audio_features,dim=0),p=2,dim=1)

        image_features,text_features = [], []
        wiki_va_emo_image, wiki_va_emo_title, wiki_va_emo_combined, wiki_va_emo_desc=[], [], [], []
        for wiki_batch in tqdm(wikiart_loader):
            categories = wiki_batch['category']
            images = wiki_batch["images"].to(device)        # Tensor shape: [B, 3, H, W]
            titles = wiki_batch["titles"]        # List of titles
            years = wiki_batch["years"]          # List of years
            artists = wiki_batch["artists"]      # List of artists

            wiki_va_emo_image.append(torch.cat([wiki_batch['emotion_valence_image_only'],wiki_batch['emotion_arousal_image_only']],dim=1))
            wiki_va_emo_combined.append(torch.cat([wiki_batch['emotion_valence_combined'],wiki_batch['emotion_arousal_combined']],dim=1))
            wiki_va_emo_title.append(torch.cat([wiki_batch['emotion_valence_title_only'],wiki_batch['emotion_arousal_title_only']],dim=1))
            wiki_va_emo_desc.append(torch.cat([wiki_batch['valence_description'],wiki_batch['arousal_description']],dim=1))

            texts=[[f'The painting {title} is, an art of {category}, created by {artist} in {year}'] for (title,category,artist,year) in zip(titles,categories,artists,years)]
            ((_, image_embeddings, _), _), _ = model(image=images)
            image_features.append(image_embeddings)

            ((_, _, text_embeddings), _), _ = model(text=texts)
            text_features.append(text_embeddings)
        
        wiki_va_emo_image=torch.cat(wiki_va_emo_image,dim=0)
        wiki_va_emo_desc=torch.cat(wiki_va_emo_desc,dim=0)
        wiki_va_emo_combined=torch.cat(wiki_va_emo_combined,dim=0)
        wiki_va_emo_title=torch.cat(wiki_va_emo_title,dim=0)
        
        image_features=F.normalize(torch.cat(image_features,dim=0),p=2,dim=1)
        text_features=F.normalize(torch.cat(text_features,dim=0),p=2,dim=1)

        #building pairs
        for emotion_va_used in [wiki_va_emo_image, wiki_va_emo_title, wiki_va_emo_combined, wiki_va_emo_desc]: # choose one of the emotion variants: image, title, combined, description
            va_dist_art2audio = pairwise_euclidean_distance(emotion_va_used, deam_train_va)
            results = get_metrics_(pairwise_cosine_similarity(image_features, audio_features), va_dist_art2audio, TOP_K)
            print(f'Recall@{TOP_K}: {results:.2f}')
            # vals,ids=torch.topk(va_dist_art2audio,k=1,dim=1,largest=False)
            # print(f'Paired audio with top-1 audios for each image-text pair: {ids.flatten()}')
            # print(f'Ids has {ids.flatten().size()} while {len(set(ids.flatten().tolist()))} unique ids')

            # # Remove duplicates
            # # This is to ensure that we only keep unique audio ids and their corresponding wikiart ids
            # # This is necessary because the same audio can be paired with multiple images or texts
            # # We will keep the first occurrence of each audio id
            # # and the corresponding wikiart id
            # unique_ids_audio = []
            # unique_ids_wiki = []
            # for i,idx in enumerate(ids.flatten().tolist()):
            #     if idx not in unique_ids_audio:
            #         unique_ids_audio.append(idx)
            #         unique_ids_wiki.append(i)
            # print(f'Unique audio ids: {len(unique_ids_audio)}')

            # audio_features_sub=audio_features[unique_ids_audio,:]
            # image_features_sub=image_features[unique_ids_wiki,:]
            # text_features_sub=text_features[unique_ids_wiki,:]

            # recall_i2a=get_metrics(image_features_sub,audio_features_sub,TOP_K)
            # recall_t2a=get_metrics(text_features_sub,audio_features_sub,TOP_K)

            # print(f'Recall@{TOP_K}: image-to-audio {recall_i2a:.2f}: text-to-audio {recall_t2a:.2f}')
        
            # #image+text---->audio
            # print(f'Using image+text pairs to retrieve audio')
            # added_features=F.normalize(image_features_sub+text_features_sub,p=2,dim=1)
            # recall_added=get_metrics(added_features,audio_features_sub,TOP_K)
            # print(f'Recall@{TOP_K}: {recall_added:.2f} when using add(image,text)---->audio')
            
            # #image*text---->audio
            # # Note: This is a simple element-wise multiplication, which may not be the best approach
            # print(f'Using image*text pairs to retrieve audio')
            # multiplied_features=F.normalize(image_features_sub*text_features_sub,p=2,dim=1)
            # recall_multiplied=get_metrics(multiplied_features,audio_features_sub,TOP_K)
            # print(f'Recall@{TOP_K}: {recall_multiplied:.2f} when using multiply(image,text)---->audio')

            # #attention-based retrieval
            # # print(f'Using attention-based retrieval')
            # # attention_features = F.normalize(self_attention(image_features, text_features), p=2, dim=1)
            # # recall_attention = get_metrics(attention_features, audio_features, TOP_K)

            print('\n' + '='*50 + '\n')
    
if __name__=='__main__':
    torch.manual_seed(0)

    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    model = AudioCLIP(pretrained=f'./AudioCLIP/assets/{MODEL_FILENAME}')
    extract_features(model)
