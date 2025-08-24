import torch

def compute_map_at_k(similarity_matrix, relevance_matrix, k=10):
    """
    Compute the mean Average Precision at K (mAP@K) for image-to-text retrieval.
    
    Args:
        similarity_matrix (torch.Tensor): Matrix of similarity scores between query images and texts.
                                          Shape: (n_queries, n_texts)
        relevance_matrix (torch.Tensor): Binary matrix indicating relevant text for each query image.
                                         Shape: (n_queries, n_texts)
        k (int): Number of top results to consider.
    
    Returns:
        float: mAP@K score
    """
    n_queries = similarity_matrix.shape[0]
    aps = []  # List to store AP@K for each query

    for i in range(n_queries):
        # Get similarity scores and relevance for current query
        sim_scores = similarity_matrix[i]
        rel_labels = relevance_matrix[i]
        
        # Sort indices by similarity scores in descending order
        _, ranked_indices = torch.sort(sim_scores, descending=True)
        
        # Consider only top-k results
        top_k_indices = ranked_indices[:k]
        top_k_relevance = rel_labels[top_k_indices].bool()
        
        # Calculate number of relevant texts in top-k
        n_relevant = torch.sum(top_k_relevance).item()
        
        if n_relevant == 0:
            # If no relevant texts in top-k, AP is 0
            aps.append(0.0)
            continue
        
        # Calculate precision at each rank where text is relevant
        relevant_ranks = torch.where(top_k_relevance)[0] + 1  # Convert to 1-indexed ranks
        precisions = []
        for idx, rank in enumerate(relevant_ranks):
            # Precision at rank: (number of relevant texts up to rank) / rank
            precisions.append((idx + 1) / rank.item())
        
        # Average precision is the mean of precisions at relevant ranks
        ap = torch.mean(torch.tensor(precisions)).item()
        aps.append(ap)
    
    # mAP@K is the mean of APs across all queries
    return torch.mean(torch.tensor(aps)).item()

def compute_recall_at_k(similarity_matrix, relevance_matrix, k=10):
    """
    Compute Recall at K for image-to-text retrieval.
    
    Args:
        similarity_matrix (torch.Tensor): Matrix of similarity scores between query images and texts.
                                          Shape: (n_queries, n_texts)
        relevance_matrix (torch.Tensor): Binary matrix indicating relevant text for each query image.
                                         Shape: (n_queries, n_texts)
        k (int): Number of top results to consider.
    
    Returns:
        float: Recall@K score
    """
    n_queries = similarity_matrix.shape[0]
    recalls = []  # List to store Recall@K for each query

    for i in range(n_queries):
        # Get similarity scores and relevance for current query
        sim_scores = similarity_matrix[i]
        rel_labels = relevance_matrix[i]
        
        # Sort indices by similarity scores in descending order
        _, ranked_indices = torch.sort(sim_scores, descending=True)
        
        # Consider only top-k results
        top_k_indices = ranked_indices[:k]
        top_k_relevance = rel_labels[top_k_indices].bool()
        
        # Calculate number of relevant texts in top-k
        n_relevant_in_top_k = torch.sum(top_k_relevance).item()
        
        # Calculate total number of relevant texts for this query
        n_total_relevant = torch.sum(rel_labels).item()
        
        if n_total_relevant == 0:
            # If no relevant texts at all, skip this query
            continue
        
        # Recall@K: proportion of relevant texts found in top-k
        recall = n_relevant_in_top_k / n_total_relevant
        recalls.append(recall)
    
    # Average recall across all queries
    return torch.mean(torch.tensor(recalls)).item() if recalls else 0.0

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate example data
    n_queries = 100
    n_texts = 1000
    k = 10
    
    # Random similarity scores (replace with actual model outputs)
    similarity_matrix = torch.rand(n_queries, n_texts, device=device)
    
    # Create dummy relevance matrix (5 relevant texts per query)
    relevance_matrix = torch.zeros(n_queries, n_texts, device=device)
    for i in range(n_queries):
        relevant_indices = torch.randperm(n_texts, device=device)[:5]
        relevance_matrix[i, relevant_indices] = 1
    
    # Compute mAP@K
    mapk = compute_map_at_k(similarity_matrix, relevance_matrix, k)
    print(f"mAP@{k}: {mapk:.4f}")
    
    # Compute Recall@K
    recallk = compute_recall_at_k(similarity_matrix, relevance_matrix, k)
    print(f"Recall@{k}: {recallk:.4f}")