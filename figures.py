from scipy import stats
import torchvision
import json
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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
def show_tensor_images(images_tensor, num_images=8, file_path="output_image_grid.png"):
    # Make a grid from batch
    img_grid = torchvision.utils.make_grid(images_tensor[:num_images], nrow=4)
    
    # Convert to numpy for plotting
    img_grid = img_grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_llm_ablation_metrics(res_list_of_dict, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['mAP']*3, file_path="llm_ablation.pdf"):
    fig, axes = plt.subplots(1, len(xlabels), figsize=(len(xlabels)*5, 4),sharey=True)
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
        axes[i].fill_between(x_vals, metric_vals - ci, metric_vals + ci, alpha=0.2, color=colors[i], label='95% CI')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_df_ablation_metrics(res_yes, res_no, xlabels, ylabels, file_path):
    fig, axes = plt.subplots(1, len(xlabels), figsize=(5*len(xlabels), 4), sharey=True)
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


if __name__ == "__main__":
    # with open('ablation_df_yes_results.json', 'r') as f:
    #     df_yes = json.load(f)
    # with open('ablation_df_no_results.json', 'r') as f:
    #     df_no = json.load(f)
    # ylabels=['mAP']*3
    # file_path="df_ablation.pdf"

    ##plot df ablation with multiple seeds for CIRCO
    res_df_circo_yes = []
    res_df_circo_no = []
    for seed in [0, 10, 42]:
        with open(f'./data/ablation_sdxl_img2img_circo_val_openclip_yes_results_{seed}.json', 'r') as f:
            res_df_circo_yes.append(json.load(f))
        with open(f'./data/ablation_sdxl_img2img_circo_val_openclip_no_results_{seed}.json', 'r') as f:
            res_df_circo_no.append(json.load(f))
    file_path=f"sdxl_ablation_img2img_circo_val_openclip.pdf"
    plot_df_ablation_metrics(res_df_circo_yes, res_df_circo_no, ['Guidance Scale', 'Num Infer Steps', 'Image Guidance Scale'], ['mAP','',''], file_path)

    ##plot llm ablation with multiple seeds for FASHIONIQ
    res_list_fashioniq = []
    for seed in [0, 10, 42]:
        with open(f'./data/ablation_qwen_txt2img_fashioniq_val_openclip_results_{seed}.json', 'r') as f:
            res_list_fashioniq.append(json.load(f))
    plot_llm_ablation_metrics(res_list_fashioniq, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['Recall','',''], file_path="llm_ablation_txt2img_fashioniq_val_openclip.pdf")

    ##plot llm ablation with multiple seeds for CIRCO
    res_list_circo = []
    for seed in [0, 10, 42]:
        with open(f'./data/ablation_qwen_txt2img_circo_val_openclip_results_{seed}.json', 'r') as f:
            res_list_circo.append(json.load(f))
    plot_llm_ablation_metrics(res_list_circo, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['mAP','',''], file_path="llm_ablation_txt2img_circo_val_openclip.pdf")