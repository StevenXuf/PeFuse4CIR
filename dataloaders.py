from configuration import get_default_config
from refinedfashioniq import get_refined_fashioniq_loader, transform_image
from circo import get_circo_loader
from cirr import get_cirr_loader
from fashioniq import get_fashioniq_loader

def get_dataloader(cfg, split='val', transform=None):

    dataset_name = cfg['GENERAL']['DATASET'].lower()
    extractor_name = cfg['GENERAL']['EXTRACTOR']
    if transform == None:
        transform=transform_image(cfg[extractor_name]['IMAGE_SIZE'],
                                cfg[extractor_name]['IMAGE_MEAN'],
                                cfg[extractor_name]['IMAGE_STD'])
    
    if dataset_name.lower() == "refinedfashioniq":
        dataloader = get_refined_fashioniq_loader(
            output_dir=cfg['RefinedFashionIQ']['OUTPUT_DIR'],
            batch_size=cfg['GENERAL']['BATCH_SIZE'],
            transform=transform
        )
    elif dataset_name.lower() == "fashioniq": #### fix here
        dataloader = get_fashioniq_loader(
            batch_size=cfg['GENERAL']['BATCH_SIZE'],
            split=split, #use val or test split
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=transform
        )
    elif dataset_name.lower() == "circo":
        dataloader = get_circo_loader(
            batch_size=cfg['GENERAL']['BATCH_SIZE'],
            split=split, #use val or test split
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=transform
        )
    elif dataset_name.lower() == "cirr": #####fix here
        dataloader = get_cirr_loader(
            batch_size=cfg['GENERAL']['BATCH_SIZE'],
            split=split, # use val or test split
            num_workers=cfg['GENERAL']['NUM_WORKERS'],
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataloader

if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    dataloader = get_dataloader(cfg)