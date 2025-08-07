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
