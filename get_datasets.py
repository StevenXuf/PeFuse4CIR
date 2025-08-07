import torch
import pandas as pd
import requests
import fire
import torchaudio

from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import to_tensor
from torchmetrics.functional.pairwise import pairwise_euclidean_distance

from text_to_vad import get_vad,get_lexicon
from configuration import get_default_config

def get_deam_loaders(threshold_valence=1.75,threshold_arousal=1.0,batch_size=64):
    ds = load_dataset("Rehead/DEAM_stripped_vocals")
    train,test=ds['train'],ds['test']
    filtered_train=filter_dataset(train,threshold_valence=threshold_valence,threshold_arousal=threshold_arousal)
    filtered_test=filter_dataset(test,threshold_valence=threshold_valence,threshold_arousal=threshold_arousal)
    return get_dataloader(filtered_train,deam_collate_fn,batch_size),get_dataloader(filtered_test,deam_collate_fn,batch_size)

def filter_dataset(dataset,threshold_valence=1.75,threshold_arousal=1.0):
    # Assuming `dataset` is your Hugging Face Dataset object
    # Filter the dataset
    filtered_dataset = dataset.filter(
        lambda example: example['valence_std'] < threshold_valence and example['arousal_std'] < threshold_arousal)

    # Set format for non-audio columns to PyTorch tensors
    non_audio_cols = [col for col in filtered_dataset.column_names if col != 'audio']
    filtered_dataset.set_format(type='torch', columns=non_audio_cols, output_all_columns=True)

    return filtered_dataset
    

    # Custom collate function to process audio
def deam_collate_fn(batch):
    audio_data = [item.pop('audio') for item in batch]  # Extract audio
    collated_batch = default_collate(batch)  # Collate non-audio features
    resample_transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=44100)
    # Convert audio arrays to tensors and add to batch
    audio_arrays = [torch.as_tensor(audio['array'],dtype=torch.float32) for audio in audio_data]

    audio_arrays=[resample_transform(waveform) for waveform in audio_arrays]
    audio_arrays=[audio/audio.abs().max() for audio in audio_arrays]

    collated_batch['audio']=pad_sequence(audio_arrays, batch_first=True, padding_value=0.0)[:,:44100*15]  # Keep only the first 5 seconds
    return collated_batch

def get_dataloader(dataset,collate_fn,batch_size=64):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Adjust batch size as needed
        shuffle=False,    # Optional: Shuffle for training
        collate_fn=collate_fn
    )
    return dataloader

def get_img_transform(IMAGE_SIZE=224):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),  # Convert to tensor [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


#processing art dataset below
class ArtworkDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image from URL
        try:
            response = requests.get(row['Image URL'], timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error loading image for ID {row['ID']}: {e}")
            # Return placeholder image on failure
            img = Image.new('RGB', (IMAGE_SIZE,IMAGE_SIZE), (128, 128, 128))

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Process metadata
        category = str(row['Category']) if pd.notna(row['Category']) else "Unknown"
        title = str(row['Title']) if pd.notna(row['Title']) else "Unknown"
        year = str(row['Year']) if pd.notna(row['Year']) else "Unknown"
        artist = str(row['Artist']) if pd.notna(row['Artist']) else "Unknown"
        emotion_valence_image_only=torch.tensor([row['emotion_valence_image_only']])
        emotion_arousal_image_only=torch.tensor([row['emotion_arousal_image_only']])
        emotion_valence_title_only,emotion_arousal_title_only,emotion_valence_combined,emotion_arousal_combined=list(map(lambda x: torch.tensor([x]),[row['emotion_valence_title_only'],row['emotion_arousal_title_only'],row['emotion_valence_combined'],row['emotion_arousal_combined']]))
        valence_description=torch.tensor([row['valence_description']])
        arousal_description=torch.tensor([row['arousal_description']])

        return category, img, title, year, artist, emotion_valence_image_only, emotion_arousal_image_only, emotion_valence_title_only,emotion_arousal_title_only,emotion_valence_combined,emotion_arousal_combined, valence_description, arousal_description

# Custom collate function to handle text metadata
def custom_art_collate(batch):
    category, images, titles, years, artists, valence_emo_image, arousal_emo_image, valence_emo_title, arousal_emo_title,  valence_emo_combined, arousal_emo_combined, valence_desc, arousal_desc= zip(*batch)

    # Stack images into tensor
    images = torch.stack(images)
    
    valence_emo_image, arousal_emo_image, valence_emo_title, arousal_emo_title,  valence_emo_combined, arousal_emo_combined, valence_desc, arousal_desc=list(map(lambda x: torch.stack(x),[valence_emo_image, arousal_emo_image, valence_emo_title, arousal_emo_title,  valence_emo_combined, arousal_emo_combined, valence_desc, arousal_desc]))
    # Return metadata as lists
    return {
        "category": category,
        "images": images,
        "titles": titles,
        "years": years,
        "artists": artists,
        "emotion_valence_image_only": valence_emo_image,
        "emotion_arousal_image_only": arousal_emo_image,
        "emotion_valence_title_only": valence_emo_title,
        "emotion_arousal_title_only": arousal_emo_title,
        "emotion_valence_combined": valence_emo_combined,
        "emotion_arousal_combined": arousal_emo_combined,
        "valence_description":valence_desc,
        "arousal_description":arousal_desc
    }


# Load and prepare data
def create_wikiart_dataloader(df,custom_art_collate,BATCH_SIZE=64,IMAGE_SIZE=224):
    # Create dataset and dataloader
    dataset = ArtworkDataset(df, transform=get_img_transform(IMAGE_SIZE))

    return get_dataloader(dataset,custom_art_collate,BATCH_SIZE)

def get_image_and_title_emotion(emotion_file,lexicon):
    df=pd.read_csv(emotion_file,sep='\t')
    #cols_to_keep=[col for col in df.columns if 'Art (image+title)' in col]
    #df=df[cols_to_keep] 
    df['emotion_valence_image_only'],df['emotion_arousal_image_only']=None,None
    df['emotion_valence_title_only'],df['emotion_arousal_title_only']=None,None
    df['emotion_valence_combined'],df['emotion_arousal_combined']=None,None
    df['valence_description'],df['arousal_description']=None,None

    image_cols = [col for col in df.columns if col.startswith("ImageOnly:")]
    title_cols = [col for col in df.columns if col.startswith("TitleOnly:")]
    image_and_title_cols=[col for col in df.columns if col.startswith("Art (image+title):")]

    for idx, row in df.iterrows():
        image_emotions=[item.split(':')[1].strip() for item in list(row[image_cols].index[row[image_cols]== 1])]
        va_emotion_image=get_vad(' '.join(image_emotions),lexicon)
        df.at[idx,'emotion_valence_image_only'],df.at[idx,'emotion_arousal_image_only']=va_emotion_image['valence'],va_emotion_image['arousal']

        title_emotions=[item.split(':')[1].strip() for item in list(row[title_cols].index[row[title_cols]== 1])]
        va_emotion_title=get_vad(' '.join(title_emotions),lexicon)
        df.at[idx,'emotion_valence_title_only'],df.at[idx,'emotion_arousal_title_only']=va_emotion_title['valence'],va_emotion_title['arousal']
        
        combined_emotions=[item.split(':')[1].strip() for item in list(row[image_and_title_cols].index[row[image_and_title_cols]== 1])]
        va_emotion_combined=get_vad(' '.join(combined_emotions),lexicon)
        df.at[idx,'emotion_valence_combined'],df.at[idx,'emotion_arousal_combined']=va_emotion_combined['valence'],va_emotion_combined['arousal']

        description=f"The painting {row['Title']} is, an art of {row['Category']}, created by {row['Artist']} in {row['Year']}"
        va_description=get_vad(description,lexicon)
        df.at[idx,'valence_description'],df.at[idx,'arousal_description']=va_description['valence'],va_description['arousal']
    
    return df

def get_shared_info(df1,df2):
    df_common = pd.merge(df1,df2,on=['ID', 'Category', 'Artist', 'Title', 'Year'],how='inner')
    return df_common

def get_all_loaders(config='./config.yaml',BATCH_SIZE=None,TSV_PATH=None,IMAGE_SIZE=None,EMOTION_PATH=None, LEXICON_PATH=None,THRESHOLD_VALENCE=None,THRESHOLD_AROUSAL=None,TOP_K=None):
    
    cfg=get_default_config('./config.yaml')

    if BATCH_SIZE is None:
        BATCH_SIZE=cfg['General']['BATCH_SIZE']
    if TSV_PATH is None:
        TSV_PATH=cfg['WikiArt']['TSV_PATH']
    if IMAGE_SIZE is None:
        IMAGE_SIZE=cfg['WikiArt']['IMAGE_SIZE']
    if EMOTION_PATH is None:
        EMOTION_PATH=cfg['WikiArt']['EMOTION_PATH']
    if LEXICON_PATH is None:
        LEXICON_PATH=cfg['Lexicon']['LEXICON_PATH']
    if THRESHOLD_VALENCE is None:
        THRESHOLD_VALENCE=cfg['DEAM']['THRESHOLD_VALENCE']
    if THRESHOLD_AROUSAL is None:
        THRESHOLD_AROUSAL=cfg['DEAM']['THRESHOLD_AROUSAL']
    if TOP_K is None:
        TOP_K=cfg['General']['TOP_K']

    lexicon=get_lexicon(LEXICON_PATH)
    df_emo=get_image_and_title_emotion(EMOTION_PATH,lexicon)
    df_main=pd.read_csv(TSV_PATH,sep='\t')
    df_main=df_main.dropna(subset=['Image URL'])
    df=get_shared_info(df_main,df_emo)

    wikiart_loader = create_wikiart_dataloader(df,custom_art_collate,BATCH_SIZE,IMAGE_SIZE)

    train_loader,test_loader=get_deam_loaders(THRESHOLD_VALENCE,THRESHOLD_AROUSAL,BATCH_SIZE)

    return {'wikiart':wikiart_loader,
            'deam_train':train_loader,
            'deam_test':test_loader
            }


if __name__ == "__main__":
    torch.manual_seed(0)
    fire.Fire(get_all_loaders)
