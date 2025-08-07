import torch
import os
import sys
import torchaudio
import requests
import io

import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalPrecision,RetrievalRecall
from transformers import AutoProcessor, ClapModel, AutoTokenizer,AutoFeatureExtractor

from token_truncation import truncate_to_desired_tokens

sys.path.append(os.path.abspath("AudioCLIP"))
from model import AudioCLIP

# sys.path.append(os.path.abspath('ImageBind'))
# from imagebind import data
# from imagebind.models import imagebind_model
# from imagebind.models.imagebind_model import ModalityType

sys.path.append(os.path.abspath('localized-narratives'))
import localized_narratives

class Flickr30K():
    def __init__(self,BATCH_SIZE=64,IMAGE_SIZE=224,MEAN=[0.48145466, 0.4578275, 0.40821073],STD=[0.26862954, 0.26130258, 0.27577711]):
        self.n_workers=os.cpu_count()
        self.IMAGE_SIZE=IMAGE_SIZE
        self.MEAN=MEAN
        self.STD=STD
        self.BATCH_SIZE=BATCH_SIZE
        flickr30k_dataset= load_dataset('nlphuji/flickr30k', cache_dir='./data',split='test')

        self.transform = Compose([
            Resize(self.IMAGE_SIZE, interpolation=Image.BICUBIC),
            CenterCrop(self.IMAGE_SIZE),
            ToTensor(),
            Normalize(mean=self.MEAN,std=self.STD)
        ])

        self.dataset = flickr30k_dataset.map(
            self.process_examples,
            batched=True,
            batch_size=self.BATCH_SIZE,
            #num_proc=self.n_workers,
            remove_columns=['img_id', 'sentids','split']
        )
    def process_examples(self,examples):
        # Process images
        examples['image'] = [self.transform(img.convert('RGB')) for img in examples['image']]

        return {'image': examples['image'],
        'caption': examples['caption'],
        'filename': examples['filename']
        }

    # 5. Collate Function with Random Caption Selection
    def collate_fn(self,batch):
        images = []
        captions = []

        for item in batch:
            images.append(item['image'])
            # Randomly select one caption from five
            idx = torch.randint(0, 5, (1,)).item()
            captions.append(item['caption'][idx])

        return {
            'image': torch.stack(list(map(torch.tensor,images))),
            'caption': captions
        }

    
    def get_loaders(self):
        loader=DataLoader(
        self.dataset,
        batch_size=self.BATCH_SIZE,
        shuffle=True,
        collate_fn=self.collate_fn,
        pin_memory=True)

        return loader

def get_samples_by_ids(dataset, id_list, id_field='filename', id_transform=None):
    """
    Efficiently retrieve samples by ID without building a full dictionary
    
    Args:
        dataset: Iterable of data samples
        id_list: Ordered list of IDs to retrieve
        id_field: Key name containing the ID in each sample
        id_transform: Function to transform IDs for matching (e.g., add extension)
    
    Returns:
        List of samples in the same order as id_list
    """
    # Create set for fast lookups
    required_ids = set(id_list)
    
    # Create mapping only for required IDs
    id_to_sample = {}
    for sample in tqdm(dataset):
        sample_id = sample[id_field]
        # Apply transformation if needed (e.g., add .jpg)
        lookup_id = id_transform(sample_id) if id_transform else sample_id

        if lookup_id in required_ids:
            # Store original ID for later ordering
            id_to_sample[lookup_id] = sample['image']
            
            # Early exit if we've found all required samples
            if len(id_to_sample) == len(required_ids):
                break
    
    # Build result list in original ID order
    res=[id_to_sample.get(id_,None) for id_ in id_list]
    return res

def collate_fn(batch):
    """Handle variable-length audio tensors and None values"""
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    audios = [item['audio'] for item in batch]
    audio_bytes = [item['audio_byte'] for item in batch]

    images = torch.stack(list(map(torch.tensor,images)))
    audios = pad_sequence(
        list(map(lambda x: torch.tensor(x).T,audios)), 
        batch_first=True,
        padding_value=0
    ).transpose(1,2)
    
    return {
        'image': images,
        'caption': captions,
        'audio': audios,
        'audio_byte': audio_bytes
    }

def get_triplets(dataset_name,anno_name,new_sample_rate=44100):
    local_dir = './data'
    store_path=f'{local_dir}/{dataset_name}_triplets'
    if os.path.exists(store_path):
        triplet_dataset=Dataset.load_from_disk(store_path)
    else:
        data_loader = localized_narratives.DataLoader(local_dir)
        data_loader.download_annotations(anno_name)
        annotations = iter(data_loader.load_annotations(anno_name))

        if dataset_name=='flickr30k':
            dataset=Flickr30K().dataset
        elif dataset_name=='coco':
            dataset=COCO().dataset
        elif dataset_name=='ade20k':
            dataset=ADE20K().dataset
        elif dataset_name=='open-images':
            dataset=OpenImages().dataset

        results={'image_id':[],'caption':[],'audio':[],'audio_byte':[]}
        for loc_narr in tqdm(annotations):
            results['image_id'].append(loc_narr.image_id)
            results['caption'].append(loc_narr.caption)

            url=loc_narr.voice_recording_url
            response = requests.get(url)
            raw_bytes = response.content
            audio_bytes = io.BytesIO(raw_bytes)
            results['audio_byte'].append(raw_bytes)

            waveform, sample_rate = torchaudio.load(audio_bytes)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate!=new_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
                waveform = resampler(waveform)[:,:new_sample_rate*15]  # Keep only the first 15 seconds
                waveform=waveform/waveform.abs().max()
            results['audio'].append(waveform)
        
        results['image'] = get_samples_by_ids(
        dataset,
        id_list=results['image_id'],
        id_field='filename',
        id_transform=lambda x: x.replace('.jpg', '')  # Remove extension to match image_id
    )
        valid_indices = [i for i, img in enumerate(results['image']) if img is not None]

        results = {
            'image': [results['image'][i] for i in valid_indices],
            'caption': [results['caption'][i] for i in valid_indices],
            'audio': [results['audio'][i] for i in valid_indices],
            'audio_byte': [results['audio_byte'][i] for i in valid_indices],
        }
        triplet_dataset=Dataset.from_dict(results)
        triplet_dataset.save_to_disk(store_path)
    
    return triplet_dataset

def get_triplet_dataloader(triplet_dataset,batch_size,custom_collate_fn,num_workers=16):
    dataloader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,  # Adjust based on your GPU memory
        shuffle=False,
        num_workers=num_workers,  # Parallel loading
        pin_memory=True,  # Faster transfer to GPU
        collate_fn=custom_collate_fn  # Critical for your audio data
    )
    return dataloader


def get_features(dataloader,model,device):
    image_features_all=[]
    text_features_all=[]
    audio_features_all=[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            audio=batch['audio'].to(device)
            print(audio.size())
            image=batch['image'].to(device)
            text=[[truncate_to_desired_tokens(cap)[0]] for cap in batch['caption']]
            ((audio_features, _, _), _), _ = model(audio=audio)
            ((_, image_features, _), _), _ = model(image=image)
            ((_, _, text_features), _), _ = model(text=text)

            image_features_all.append(image_features)
            text_features_all.append(text_features)
            audio_features_all.append(audio_features)
    image_features_all,text_features_all,audio_features_all=list(map(lambda x: torch.cat(x,dim=0),[image_features_all,text_features_all,audio_features_all]))

    return image_features_all,text_features_all,audio_features_all

def compute_metrics(feat1,feat2,top_k=10):
    compute_recall=RetrievalRecall(top_k=top_k)
    compute_precision=RetrievalPrecision(top_k=top_k)

    preds=pairwise_cosine_similarity(feat1,feat2).to(feat1.device)
    targets=torch.diag(torch.ones(preds.size(0),dtype=torch.long)).to(feat1.device)
    indexes = torch.arange(preds.size(0), dtype=torch.long).unsqueeze(1).expand(*preds.size()).to(feat1.device)
    recall=compute_recall(preds,targets,indexes)
    precision=compute_precision(preds,targets,indexes)
    print(f'Recall@{top_k}: {recall:.2f}')
    print(f'Precision@{top_k}: {precision:.2f}')

if __name__=='__main__':
    triplet_dataset=get_triplets('flickr30k','flickr30k_test',new_sample_rate=48000)
    device=torch.device('cuda:1')
    batch_size=1
    top_k=10
    model_name='CLAP'  # Change to 'AudioCLIP' if you want to use AudioCLIP
    
    if model_name=='AudioCLIP':
        MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
        model = AudioCLIP(pretrained=f'./AudioCLIP/assets/{MODEL_FILENAME}').to(device)
    elif model_name=='ImageBind':
        model = imagebind_model.imagebind_huge(pretrained=True).to(device)
    elif model_name=='CLAP':
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
        feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    model.eval()
     
    dataloader=get_triplet_dataloader(triplet_dataset,batch_size,collate_fn)

    # image_features,text_features,audio_features=get_features(dataloader,model,device)

    #dataloader=Flickr30K().get_loaders()
    with torch.no_grad():
        print(f'Computing features using {dataloader.__class__.__name__}...')
        all_image_features = []
        all_text_features = []
        all_audio_features = []
        for batch in tqdm(dataloader):
            image=batch['image'].to(device)
            if model_name=='AudioCLIP':
                text=[[truncate_to_desired_tokens(cap)[0]] for cap in batch['caption']]
                ((_, image_features, _), _), _ = model(image=image)
                ((_, _, text_features), _), _ = model(text=text)
                ((audio_features, _, _), _), _ = model(audio=batch['audio'].to(device))
            elif model_name=='ImageBind':
                text=batch['caption']
                audio_byte = batch['audio_byte']
                inputs = {
                    ModalityType.TEXT: data.load_and_transform_text(text, device),
                    ModalityType.VISION: data.load_and_transform_vision_data(image),
                    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_byte, device),
                    }
                embeddings = model(inputs)
                image_features = F.normalize(embeddings[ModalityType.VISION])
                text_features = F.normalize(embeddings[ModalityType.TEXT])
                audio_features = F.normalize(embeddings[ModalityType.AUDIO])
            elif model_name=='CLAP':
                txt_inputs = tokenizer(text=batch['caption'], return_tensors="pt", padding=True).to(device)
                text_features = model.get_text_features(**txt_inputs)
                print(batch['audio'].size())
                audio_inputs = feature_extractor(batch['audio'], return_tensors="pt", sampling_rate=48000).to(device)
                audio_features = model.get_audio_features(**audio_inputs)
                # image_features = outputs.image_embeds
                # text_features = outputs.text_embeds
                # audio_features = outputs.audio_embeds
            
            # all_image_features.append(image_features)
            all_text_features.append(text_features)
            all_audio_features.append(audio_features)
        # image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        audio_features = torch.cat(all_audio_features, dim=0)
        # compute_metrics(image_features, text_features, top_k=top_k)
        # compute_metrics(text_features, image_features, top_k=top_k)
        # compute_metrics(image_features, audio_features, top_k=top_k)
        compute_metrics(text_features, audio_features, top_k=top_k)
        # compute_metrics(F.normalize(image_features + text_features), audio_features, top_k=top_k)
        # compute_metrics(F.normalize(image_features * text_features), audio_features, top_k=top_k)

    #######TODO:
    # also try to use clip model and use only audio encoder from AudioCLIP