import textwrap
import torchvision
import json
import os
import fire
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Subset
from transformers import set_seed
from diffusers import StableDiffusionXLInstructPix2PixPipeline


from retrieval import main as retrieval_main
from retrieval import load_mllm
from dataloaders import get_dataloader
from utils import get_default_config, transform_image
from prompts import get_composed_prompts
from text_to_image_and_text import generate_texts

def visualize_attention(weights, x_labels=None, y_labels=None, title='Attention Weights'):
    """
    weights: (seq_len_q, seq_len_kv)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights.cpu().detach().numpy(), cmap='viridis',
                xticklabels=x_labels, yticklabels=y_labels, square=True)
    plt.xlabel("Key / Context")
    plt.ylabel("Query")
    plt.savefig(f'{"_".join(title.split())}.png')
    plt.colorbar(label='Attention Weight')
    plt.tight_layout()

# Function to show a batch of images
def plot_llm_ablation_metrics(res_list_of_dict, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['mAP']*3, file_path="llm_ablation.pdf"):
    fig, axes = plt.subplots(1, len(xlabels), figsize=(len(xlabels)*4, 3),sharey=True)
    new_xlabels = [xlabel.replace(' ', '_') if ' ' in xlabel else xlabel for xlabel in xlabels]
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(axes)):
        x_vals = res_list_of_dict[0][f'res_{new_xlabels[i].lower()}_keys']
        metric_vals = sum([np.array(res_list_of_dict[j][f'res_{new_xlabels[i].lower()}_vals']) for j in range(len(res_list_of_dict))]) / len(res_list_of_dict)
        axes[i].plot(x_vals, metric_vals, marker=markers[i], color=colors[i])
        axes[i].set_xlabel(xlabels[i], fontsize=12, weight='bold')
        axes[i].set_ylabel(ylabels[i], fontsize=12, weight='bold')
        axes[i].grid(True, alpha=0.2)

        ci = np.std(np.array([np.array(res_list_of_dict[j][f'res_{new_xlabels[i].lower()}_vals']) for j in range(len(res_list_of_dict))]), axis=0)
        axes[i].fill_between(x_vals, metric_vals - ci, metric_vals + ci, alpha=0.2, color=colors[i], label='std')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_df_ablation_metrics(res_yes, res_no, xlabels, ylabels, file_path):
    fig, axes = plt.subplots(1, len(xlabels), figsize=(4*len(xlabels), 3), sharey=True)
    new_xlabels = [xlabel.replace(' ', '_') if ' ' in xlabel else xlabel for xlabel in xlabels]
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(axes)):
        x_vals_yes = res_yes[0][f'res_{new_xlabels[i].lower()}_keys']
        metric_vals_yes = sum([np.array(res_yes[j][f'res_{new_xlabels[i].lower()}_vals']) for j in range(len(res_yes))]) / len(res_yes)
        axes[i].plot(x_vals_yes, metric_vals_yes, marker=markers[i], color=colors[i], label='W/ MLLM')
        ci_yes = np.std(np.array([np.array(res_yes[j][f'res_{new_xlabels[i].lower()}_vals']) for j in range(len(res_yes))]), axis=0)
        axes[i].fill_between(x_vals_yes, metric_vals_yes - ci_yes, metric_vals_yes + ci_yes, alpha=0.2, color=colors[i])

        x_vals_no = res_no[0][f'res_{new_xlabels[i].lower()}_keys']
        metric_vals_no = sum([np.array(res_no[j][f'res_{new_xlabels[i].lower()}_vals']) for j in range(len(res_no))]) / len(res_no)
        axes[i].plot(x_vals_no, metric_vals_no, marker=markers[i], color=colors[i+3], linestyle='--', label='W/o MLLM')
        ci_no = np.std(np.array([np.array(res_no[j][f'res_{new_xlabels[i].lower()}_vals']) for j in range(len(res_no))]), axis=0)
        print(metric_vals_no)
        print(ci_no)
        axes[i].fill_between(x_vals_no, metric_vals_no - ci_no, metric_vals_no + ci_no, alpha=0.2, color=colors[i+3])

        axes[i].set_xlabel(xlabels[i], fontsize=12, weight='bold')
        axes[i].set_ylabel(ylabels[i], fontsize=12, weight='bold')
        axes[i].grid(True, alpha=0.2)

        axes[i].legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_retrieval_example(queries, retrieved_images, task, filename="retrieval_example.pdf"):
    """
    query_text: string
    retrieved_images: list of 5 images (numpy arrays or PIL images)
    """
    nrow = len(queries['reference_img']) * 2
    ncol = len(retrieved_images[0])
    subplot_size = 4
    fig = plt.figure(figsize=(ncol*subplot_size, nrow*subplot_size))
    
    # Use a GridSpec with two rows
    gs = fig.add_gridspec(nrow, ncol, height_ratios=[1]*nrow)

    for i in range(len(queries['reference_img'])):
        # ---- Row 1: Text spanning all 5 columns ----
        ref_img = queries['reference_img'][i]
        modification = "\n".join(textwrap.wrap(queries['relative_caption'][i], width=25))
        target_desc = queries['composed_description'][i]
        target_desc = "\n".join(textwrap.wrap(target_desc, width=25))
        ref_ax = fig.add_subplot(gs[2*i, 1])
        ref_ax.imshow(ref_img)
        ref_ax.axis('off')
        modification_ax = fig.add_subplot(gs[2*i, 2])
        modification_ax.text(0.5, 0.5, modification, ha='center', va='center', fontsize=14)
        modification_ax.axis('off')
        ax_text = fig.add_subplot(gs[2*i, 3])
        if task == 'txt2img':
            ax_text.text(
                0.5, 0.5, target_desc,
                ha='center', va='center', fontsize=14
            )
        elif task == 'img2img':
            ax_text.imshow(queries['generated_target_images'][i])
        ax_text.axis('off')

        # ref_ax.text(-0.05, 0.5,                    # Position left of axes
        #             'Reference Image',                   # This acts as your ylabel
        #             transform=ref_ax.transAxes,        # Use axes coordinates
        #             rotation=90,                    # Vertical
        #             fontsize=10,
        #             fontweight='bold',
        #             va='center',
        #             ha='center',
        #             color='darkblue')
        if i == 0:
            ref_ax.set_title('Reference Image', fontsize=10, weight='bold', color='darkblue')
            modification_ax.set_title('Modification', fontsize=10, weight='bold', color='darkblue')
            if task == 'txt2img':
                ax_text.set_title('Composed Description', fontsize=10, weight='bold', color='darkblue')
            elif task == 'img2img':
                ax_text.set_title('Synthesized Target', fontsize=10, weight='bold', color='darkblue')

        # ---- Row 2: Five retrieved images ----
        for j in range(len(retrieved_images[i])):
            ax = fig.add_subplot(gs[2*i+1, j])
            ax.imshow(retrieved_images[i][j])
            if j == 0:
                ax.set_ylabel(f"Top-5 Results", fontsize=10, fontweight='bold', color='darkblue')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    # Remove all gaps between image axes
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def search_failure_cases(cfg, **kwargs):
    seed = kwargs.get('seed', cfg['GENERAL']['SEED'])
    task = kwargs.get('task', cfg['GENERAL']['TASK'])
    set_seed(seed)
    kwargs['extractor'] = 'OPENCLIP'
    filename = f'failure_cases_{kwargs["extractor"].lower()}_{task}_{seed}.json'
    if os.path.exists(filename):
        failure_info = json.load(open(filename, 'r'))
    else:
        avg_map, inference_time_per_sample, failure_info = retrieval_main(cfg, **kwargs)
        with open(filename, 'w') as f:
            json.dump(failure_info, f)
    return failure_info

def plot_failure_cases(**kwargs):
    cfg = get_default_config("config.yaml")
    task = kwargs.get('task', cfg['GENERAL']['TASK'])
    split = kwargs.get('split', cfg['GENERAL']['SPLIT'])
    dataset_name = kwargs.get('dataset', cfg['GENERAL']['DATASET'])
    extractor = kwargs.get('extractor', cfg['GENERAL']['EXTRACTOR'])
    batch_size = kwargs.get('batch_size', cfg['GENERAL']['BATCH_SIZE'])
    seed = kwargs.get('seed', cfg['GENERAL']['SEED'])
    set_seed(seed)
    failure_info = search_failure_cases(cfg, **kwargs)

    n_samples = 4
    rand_idx = np.random.choice(len(failure_info['failure_id'])+1, size=n_samples, replace=False)
    query_ids = [failure_info['failure_id'][idx] for idx in rand_idx]
    retrieved_ids = [failure_info['retrieved_ids'][idx] for idx in rand_idx]

    query_loader = get_dataloader(cfg, 
                                split=split, 
                                mode='relative',
                                dataset_name=dataset_name, 
                                extractor_name=extractor,
                                batch_size=batch_size,
                                transform=None
                                )
    query_subset = Subset(query_loader.dataset, query_ids)
    ref_imgs = [item['reference_img'] for item in query_subset]
    modifications = [item['relative_caption'] for item in query_subset]

    composed_messages = {'texts':list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(ref_imgs, modifications))),
                        'images':[]
                        }
    gen_config, processor, text_generation_model = load_mllm(cfg, **kwargs)
    generated_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model)
    queries = {'reference_img': ref_imgs,
            'relative_caption': modifications,
            'composed_description': generated_descriptions
            }
    if task == 'img2img':
        device = torch.device(f"cuda:{kwargs.get('device')}") if kwargs.get('device') is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_gen_model_name = kwargs.get('image_gen_model', cfg['GENERAL']['IMAGE_GEN_MODEL']).upper()
        image_size = cfg['IMAGE-GENERATION'][image_gen_model_name]['IMAGE_SIZE']
        n_infer_steps = kwargs.get('n_infer_steps', cfg['IMAGE-GENERATION'][image_gen_model_name]['NUM_INFERENCE_STEPS'])
        image_guidance_scale = kwargs.get('image_guidance_scale', cfg['IMAGE-GENERATION'][image_gen_model_name]['IMAGE_GUIDANCE_SCALE'])
        guidance_scale = kwargs.get('guidance_scale', cfg['IMAGE-GENERATION'][image_gen_model_name]['GUIDANCE_SCALE'])
        image_generation_model = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(cfg['IMAGE-GENERATION'][image_gen_model_name]['MODEL_NAME'], 
                                                                                            torch_dtype=torch.float16).to(device)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        img_transform_for_generation = transform_image(image_size)
        ref_imgs = torch.stack([img_transform_for_generation(pil_img) for pil_img in ref_imgs]).to(device)
        generated_target_images = image_generation_model(
            prompt=generated_descriptions,
            image=ref_imgs,
            width=image_size,
            height=image_size,
            num_inference_steps=n_infer_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator
            ).images
        queries['generated_target_images'] = generated_target_images

    target_loader = get_dataloader(cfg, 
                                dataset_name=dataset_name, 
                                split='val', 
                                mode='classic',
                                batch_size=batch_size, 
                                extractor_name=extractor
                                )
    targets = [[item['target_image'] for item in Subset(target_loader.dataset, group)] for group in retrieved_ids]
    
    plot_retrieval_example(queries, targets, task=task, filename=f'{task}_failure_cases_{dataset_name}_{extractor}_{seed}.pdf')

    

if __name__ == "__main__":
    # with open('ablation_df_yes_results.json', 'r') as f:
    #     df_yes = json.load(f)
    # with open('ablation_df_no_results.json', 'r') as f:
    #     df_no = json.load(f)
    # ylabels=['mAP']*3
    # file_path="df_ablation.pdf"

    ##plot df ablation with multiple seeds for CIRCO
    # res_df_circo_yes = []
    # res_df_circo_no = []
    # for seed in [0, 10, 42]:
    #     with open(f'./data/ablation_sdxl_img2img_circo_val_openclip_yes_results_{seed}.json', 'r') as f:
    #         res_df_circo_yes.append(json.load(f))
    #     with open(f'./data/ablation_sdxl_img2img_circo_val_openclip_no_results_{seed}.json', 'r') as f:
    #         res_df_circo_no.append(json.load(f))
    # file_path=f"sdxl_ablation_img2img_circo_val_openclip.pdf"
    # plot_df_ablation_metrics(res_df_circo_yes, res_df_circo_no, ['Guidance Scale', 'Num Infer Steps', 'Image Guidance Scale'], ['mAP','',''], file_path)

    ##plot llm ablation with multiple seeds for FASHIONIQ
    # res_list_fashioniq = []
    # for seed in [0, 10, 42]:
    #     with open(f'./data/ablation_qwen_txt2img_fashioniq_val_openclip_results_{seed}.json', 'r') as f:
    #         res_list_fashioniq.append(json.load(f))
    # plot_llm_ablation_metrics(res_list_fashioniq, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['Recall','',''], file_path="llm_ablation_txt2img_fashioniq_val_openclip.pdf")

    ##plot llm ablation with multiple seeds for CIRCO
    # res_list_circo = []
    # for seed in [0, 10, 42]:
    #     with open(f'./data/ablation_qwen_txt2img_circo_val_openclip_results_{seed}.json', 'r') as f:
    #         res_list_circo.append(json.load(f))
    # plot_llm_ablation_metrics(res_list_circo, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['mAP','',''], file_path="llm_ablation_txt2img_circo_val_openclip.pdf")

    fire.Fire(plot_failure_cases)