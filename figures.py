import torchvision

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

def plot_ablation_metrics(res_dict, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['mAP']*3, file_path="llm_ablation.pdf"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    new_xlabels = [xlabel.replace(' ', '_') if ' ' in xlabel else xlabel for xlabel in xlabels]
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(axes)):
        x_vals = res_dict[f'res_{new_xlabels[i].lower()}_keys']
        metric_vals = res_dict[f'res_{new_xlabels[i].lower()}_vals']
        axes[i].plot(x_vals, metric_vals, marker=markers[i], color=colors[i])
        axes[i].set_xticks(x_vals)
        axes[i].set_xlabel(xlabels[i], fontsize=12)
        axes[i].set_ylabel(ylabels[i], fontsize=12)
        axes[i].grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

if __name__ == "__main__":
    res = {
        'res_guidance_scale_keys': [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        'res_guidance_scale_vals': [0.1 * i for i in range(1, 8)],
        'res_num_infer_steps_keys': [10, 20, 30, 50, 100],
        'res_num_infer_steps_vals': [0.2 * i for i in range(1, 6)],
        'res_image_guidance_scale_keys': [1.0, 1.3, 1.7, 2.0, 3.0, 4.0, 5.0],
        'res_image_guidance_scale_vals': [0.3 * i for i in range(1, 8)]
    }
    plot_ablation_metrics(res, xlabels=['Guidance Scale', 'Num Infer Steps', 'Image Guidance Scale'], ylabels=['mAP']*3, file_path="df_ablation.pdf")
    