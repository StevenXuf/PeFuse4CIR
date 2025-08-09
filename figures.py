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
