import torch
import torch.nn.functional as F

def self_attention_batched(x):
    """
    x: (batch_size, seq_len, dim)
    Returns:
        attended: (batch_size, seq_len, dim)
        weights: (batch_size, seq_len, seq_len)
    """
    B, L, D = x.shape
    # Compute attention scores (dot product): (B, L, L)
    attn_scores = torch.bmm(x, x.transpose(1, 2))
    attn_weights = F.softmax(attn_scores, dim=-1)
    attended = torch.bmm(attn_weights, x)  # (B, L, D)
    return attended, attn_weights

def cross_attention_batched(query, context):
    """
    query: (batch_size, len_q, dim)
    context: (batch_size, len_kv, dim)

    Returns:
        attended: (batch_size, len_q, dim)
        weights: (batch_size, len_q, len_kv)
    """
    B, Lq, D = query.shape
    Lkv = context.size(1)
    attn_scores = torch.bmm(query, context.transpose(1, 2))  # (B, Lq, Lkv)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attended = torch.bmm(attn_weights, context)  # (B, Lq, D)
    return attended, attn_weights

def co_attention_batched(text_feats, image_feats):
    """
    text_feats: (batch_size, len_text, dim)
    image_feats: (batch_size, len_img, dim)

    Returns:
        text_attended: (batch_size, len_text, dim)
        image_attended: (batch_size, len_img, dim)
        text_weights: (batch_size, len_text, len_img)
        image_weights: (batch_size, len_img, len_text)
    """
    text_attended, text_weights = cross_attention_batched(text_feats, image_feats)
    image_attended, image_weights = cross_attention_batched(image_feats, text_feats)
    return text_attended, image_attended, text_weights, image_weights



