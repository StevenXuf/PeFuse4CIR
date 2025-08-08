import torch
import torch.nn.functional as F

def self_attention(x):
    """
    x: (seq_len, dim)
    Returns:
        attended: (seq_len, dim)
        weights: (seq_len, seq_len)
    """
    # Compute attention scores (dot product): (L, L)
    attn_scores = x @ x.T
    attn_weights = F.softmax(attn_scores, dim=-1)
    attended = attn_weights @ x  # (L, D)
    return attended, attn_weights

def cross_attention(query, context):
    """
    query: (len_q, dim)
    context: (len_kv, dim)

    Returns:
        attended: (len_q, dim)
        weights: (len_q, len_kv)
    """
    attn_scores = query @ context.T  # (Lq, Lkv)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attended = attn_weights @ context  # (Lq, D)
    return attended, attn_weights

def co_attention(text_feats, image_feats):
    """
    text_feats: (len_text, dim)
    image_feats: (len_img, dim)

    Returns:
        text_attended: (len_text, dim)
        image_attended: (len_img, dim)
        text_weights: (len_text, len_img)
        image_weights: (len_img, len_text)
    """
    text_attended, text_weights = cross_attention(text_feats, image_feats)
    image_attended, image_weights = cross_attention(image_feats, text_feats)
    return text_attended, image_attended, text_weights, image_weights
