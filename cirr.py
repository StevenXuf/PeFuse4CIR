import torch
import os
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from configuration import get_default_config
from fashioniq import transform_image

class CIRRDataset(Dataset):
    def __init__(self, root_dir, split="val", captions_folder="captions", captions_ext_folder="captions_ext", transform=None):
        """
        Args:
            root_dir (str): Path to `data/cirr`.
            split (str): 'train', 'val', or 'test1'.
            captions_folder (str): 'captions' or 'captions_ext'.
            transform (callable, optional): Optional transform to be applied
                on a sample (applied to both reference and target images).
        """
        self.root_dir = root_dir
        self.split = split
        self.captions_folder = captions_folder
        self.captions_ext_folder = captions_ext_folder
        self.transform = transform

        self.captions_path = os.path.join(
            root_dir,
            captions_folder,
            f"cap.rc2.{self.split}.json"
        )
        self.captions_ext_path = os.path.join(
            root_dir,
            captions_ext_folder,
            f"cap.ext.rc2.{self.split}.json"
        )
        if not os.path.exists(self.captions_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")
        if not os.path.exists(self.captions_ext_path):
            raise FileNotFoundError(f"Captions file not found: {self.captions_ext_path}")

        with open(self.captions_path, 'r') as f:
            captions = json.load(f)
        with open(self.captions_ext_path, 'r') as p:
            captions_ext_info = json.load(p)

        self.img_root = os.path.join(root_dir, "img_raw")

        # Pre-filter to keep only samples with both images existing
        self.caption_data = {}
        if self.split == "test1":
            for entry in captions:
                ref_img_path = self._find_image_path(entry["reference"])
                if ref_img_path is not None:
                    self.caption_data[entry["pairid"]] = entry
                else:
                    print(f"Skipping missing image for pairid={entry.get('pairid')}")
        elif self.split == "val" or self.split == "train":
            for entry in captions:
                ref_img_path = self._find_image_path(entry["reference"])
                tgt_img_path = self._find_image_path(entry["target_hard"])
                if ref_img_path is not None and tgt_img_path is not None:
                    self.caption_data[entry["pairid"]] = entry
                else:
                    print(f"Skipping missing image for pairid={entry.get('pairid')}")
        else:
            raise ValueError("split should be in ['val', 'test1']")

        self.ext_captions = self._get_ext_captions(captions_ext_info)
    


    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, idx):
        current_id = list(self.caption_data.keys())[idx]
        entry = self.caption_data[current_id]
        ext_caption = self.ext_captions[current_id]

        if self.split == "test1":
            reference_id = entry["reference"]
            caption = entry["caption"].replace(".", "") + ' and ' + ext_caption
            pairid = entry["pairid"]

            ref_img_path = self._find_image_path(reference_id)
            ref_img = Image.open(ref_img_path).convert("RGB")
            # ref_img = ref_img.resize((768, 768), Image.Resampling.BICUBIC) if ref_img.size[0] > 768 or ref_img.size[1] > 768 else ref_img
            
            return {
                "reference_image": ref_img,
                "reference_id": reference_id,
                "caption": caption,
                "query_id": pairid
            }

        else:
            reference_id = entry["reference"]
            target_id = entry["target_hard"]
            caption = entry["caption"].replace(".", "") + ' and ' + ext_caption
            pairid = entry["pairid"]

            ref_img_path = self._find_image_path(reference_id)
            tgt_img_path = self._find_image_path(target_id)

            ref_img = Image.open(ref_img_path).convert("RGB")
            # ref_img = ref_img.resize((768, 768), Image.Resampling.BICUBIC) if ref_img.size[0] > 768 or ref_img.size[1] > 768 else ref_img
            tgt_img = Image.open(tgt_img_path).convert("RGB")
            # tgt_img = tgt_img.resize((768, 768), Image.Resampling.BICUBIC) if tgt_img.size[0] > 768 or tgt_img.size[1] > 768 else tgt_img

            if self.transform:
                ref_img = self.transform(ref_img)
                tgt_img = self.transform(tgt_img)

            return {
                "reference_image": ref_img,
                "target_image": tgt_img,
                "caption": caption,
                "reference_id": reference_id,
                "target_id": target_id,
                "query_id": pairid
            }

    def _find_image_path(self, img_id):
        """
        Locate an image path in the CIRR img_raw folder given its ID.
        """
        split_folder = img_id.split("-")[0]  # train/dev/test1
        img_filename = img_id + ".png"

        if split_folder == "train":
            train_dir = os.path.join(self.img_root, split_folder)
            if not os.path.exists(train_dir):
                return None
            for subfolder in os.listdir(train_dir):
                candidate = os.path.join(train_dir, subfolder, img_filename)
                if os.path.exists(candidate):
                    return candidate
        else:
            candidate = os.path.join(self.img_root, split_folder, img_filename)
            if os.path.exists(candidate):
                return candidate
        
        return None

    def _get_ext_captions(self, captions_ext):
        ext_captions = {}
        uninformative_phrases = ['covered in query', 'none existed', 'nothing worth mentioning']
        if self.split == "test1":
            for ext in captions_ext:
                ref_id = ext["reference"]
                pair_id = ext["pairid"]
                if pair_id in self.caption_data:
                    if self.caption_data[pair_id]['reference'] == ref_id:
                        filtered_caps = [cap for cap in list(map(lambda x: x.lower().replace(".", ""), list(ext['caption_extend'].values()))) if not any(phrase in cap for phrase in uninformative_phrases)]
                        ext_captions[pair_id] = ' and '.join(filtered_caps)

        else:
            for ext in captions_ext:
                ref_id = ext["reference"]
                tgt_id = ext["target_hard"]
                pair_id = ext["pairid"]
                if pair_id in self.caption_data:
                    if self.caption_data[pair_id]['reference'] == ref_id and self.caption_data[pair_id]['target_hard'] == tgt_id:
                        filtered_caps = [cap for cap in list(map(lambda x: x.lower().replace(".", ""), list(ext['caption_extend'].values()))) if not any(phrase in cap for phrase in uninformative_phrases)]
                        ext_captions[pair_id] = ' and '.join(filtered_caps)
        return ext_captions

def get_cirr_loader(data_path, batch_size=16, split='val', num_workers=0, transform=None):
    val_collate_fn = lambda batch: {
                            "reference_img": torch.stack([transform(item["reference_image"]) if transform is not None else item['reference_image'] for item in batch]),
                            "reference_pil": [item["reference_image"] for item in batch], 
                            "target_img": torch.stack([transform(item["target_image"]) if transform is not None else item['target_image'] for item in batch]),
                            "target_pil": [item["target_image"] for item in batch],
                            "all_target_img": torch.stack([transform(item["target_image"]) if transform is not None else item['target_image'] for item in batch]),
                            "all_target_pil": [item["target_image"] for item in batch],
                            "all_target_length": list(map(len,[[item["target_image"]] for item in batch])),
                            "caption": [item["caption"] for item in batch],
                            "reference_id": [item["reference_id"] for item in batch],
                            "target_id": [item["target_id"] for item in batch],
                            "query_id": [item["query_id"] for item in batch]
                            }
    test_collate_fn = lambda batch: {
        "reference_img": torch.stack([transform(item["reference_image"]) if transform is not None else item['reference_image'] for item in batch]),
        "reference_pil": [item["reference_image"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "reference_id": [item["reference_id"] for item in batch],
        "query_id": [item["query_id"] for item in batch]
    }
    dataset = CIRRDataset(
        root_dir=data_path,
        split=split,
        captions_folder="captions"
    )
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=num_workers,
                        collate_fn=val_collate_fn if split == 'val' else test_collate_fn
                        )
    return loader


if __name__ == "__main__":
    cfg = get_default_config('config.yaml')

    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )

    loader = get_cirr_loader(cfg['CIRR']['IMAGE_FOLDER'], transform=img_transform, split='test1')
    print(f"Number of samples in {cfg['CIRR']['IMAGE_FOLDER']}: {len(loader.dataset)}")
    for batch in loader:
        print(batch["reference_img"].shape)
        print(batch["caption"])
        break