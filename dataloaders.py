from refinedfashioniq import get_refined_fashioniq_loader
from circo import get_circo_loader
from cirr import get_cirr_loader
from fashioniq import get_fashioniq_loader
from utils import transform_image, get_default_config

def get_dataloader(cfg, split='val', mode='relative', transform=None, dataset_name=None, extractor_name=None, batch_size=None):

    if dataset_name is None:
        dataset_name = cfg['GENERAL']['DATASET']
    if batch_size is None:
        batch_size = cfg['GENERAL']['BATCH_SIZE']
    if extractor_name is None:
        extractor_name = cfg['GENERAL']['EXTRACTOR']

    if transform == None:
        transform=transform_image(cfg[extractor_name]['IMAGE_SIZE'],
                                cfg[extractor_name]['IMAGE_MEAN'],
                                cfg[extractor_name]['IMAGE_STD'])
    print(f"Using transform:\n {transform}")
    
    if dataset_name.lower() == "refinedfashioniq":
        dataloader = get_refined_fashioniq_loader(
            output_dir=cfg['RefinedFashionIQ']['IMAGE_FOLDER'],
            batch_size=batch_size,
            transform=transform
        )
    elif dataset_name.lower() == "fashioniq":
        dataloader = get_fashioniq_loader(
            data_path=cfg['FashionIQ']['IMAGE_FOLDER'],
            batch_size=batch_size,
            split=split, #use val or test split
            mode=mode,
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=transform
        )
    elif dataset_name.lower() == "circo":
        dataloader = get_circo_loader(
            data_path=cfg['CIRCO']['IMAGE_FOLDER'],
            batch_size=batch_size,
            split=split, #use val or test split
            mode=mode,
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=transform
        )
    elif dataset_name.lower() == "cirr":
        dataloader = get_cirr_loader(
            data_path=cfg['CIRR']['IMAGE_FOLDER'],
            batch_size=batch_size,
            split=split, # use val or test1 split
            mode=mode,
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if mode == 'relative':
        print(f'ON {mode.upper()} MODE.')
        print(f'{dataset_name.upper()} {split.upper()} LOADED SUCCESSFULLY.')
    elif mode == 'classic':
        print(f'ON {mode.upper()} MODE.')
        print(f'{dataset_name.upper()} LOADED SUCCESSFULLY.')
    print(f'{len(dataloader.dataset)} ITEMS IN DATALOADER.')

    return dataloader

if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    dataloader = get_dataloader(cfg, dataset_name='circo', split='val', mode='relative')
    print(len(dataloader.dataset))