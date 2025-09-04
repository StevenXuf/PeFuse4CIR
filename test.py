import torch

from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from torchmetrics.retrieval import RetrievalRecall

from dataloaders import get_dataloader
from utils import get_default_config
from prompts import get_composed_prompts
from text_to_image_and_text import extract_text_features, extract_image_features, generate_texts, fashioniq_eval

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
gen_config = GenerationConfig(do_sample=True,
                                    temperature=0.1,
                                    top_p=0.9,
                                    top_k=50,
                                    max_new_tokens=120
                                )
text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', 
                                                                        torch_dtype=torch.bfloat16, 
                                                                        device_map={"": device}, 
                                                                        attn_implementation='flash_attention_2'
                                                                        ).to(device)
text_generation_model.eval()
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', 
                                            padding_side='left', 
                                            use_fast=True
                                            )

cfg = get_default_config("config.yaml")
dataloader = get_dataloader(cfg, dataset_name='fashioniq', split='val', mode='relative', batch_size=64, extractor_name='OPENVISION')
test_loader = get_dataloader(cfg, dataset_name='fashioniq', split='val', mode='classic', batch_size=64, extractor_name='OPENVISION')

feature_extraction_model, preprocess = create_model_from_pretrained('hf-hub:UCSC-VLAA/openvision-vit-base-patch16-224')
tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/openvision-vit-base-patch16-224')

print(feature_extraction_model.__class__.__name__)
query_feat = []
target_feat = []
truth = []
targets = []
with torch.no_grad(), torch.amp.autocast('cuda'):
    for i,batch in enumerate(tqdm(dataloader)):
        reference_pil = batch['reference_pil']
        caption = batch['caption']
        target_pil = batch['target_pil']
        truth.extend(batch['target_id'])

        composed_messages = list(map(lambda x: get_composed_prompts('fashioniq', *x),zip(reference_pil, caption)))
        composed_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model,)
        query_feat.append(extract_text_features(composed_descriptions, 'OPENVISION', tokenizer, feature_extraction_model))
        target_feat.append(extract_image_features(target_pil, 'OPENVISION', feature_extraction_model, preprocess))

        # if i == 10:
        #     break

query_feat = torch.cat(query_feat, dim=0)
target_feat = torch.cat(target_feat, dim=0)

sim = pairwise_cosine_similarity(target_feat, query_feat)

for k in [1,5,10]:
    compute = RetrievalRecall(top_k=k)
    targets = torch.diag(torch.ones(sim.size(0),dtype=torch.long)).to(sim.device)
    indexes = torch.arange(sim.size(0), dtype=torch.long).unsqueeze(1).expand(*sim.size()).to(sim.device)
    recall = compute(sim, targets, indexes)
    print(f'Recall@{k}: {recall.item()*100:.2f}%')