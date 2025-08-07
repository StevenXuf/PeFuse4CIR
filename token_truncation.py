import re
import tiktoken

# Load the CLIP-compatible tokenizer (uses 'cl100k_base' which is similar in style)
encoding = tiktoken.get_encoding("cl100k_base")  # Can also try "p50k_base" or "r50k_base" for OpenAI-style BPE

def normalize_text(text):
    # Remove space before punctuations (e.g., "word ," -> "word,")
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text

def truncate_to_desired_tokens(text, max_tokens=72):
    text = normalize_text(text)
    
    # Tokenize and truncate
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text, len(tokens)
    
    # Truncate token-wise
    truncated_tokens = tokens[:max_tokens]
    decoded = encoding.decode(truncated_tokens)

    # Final cleanup: remove space before punctuation again, if decoding added any
    decoded = normalize_text(decoded)
    token_count = len(encoding.encode(decoded))
    
    return decoded, token_count

# Example usage
if __name__ == "__main__":
    text = "A cat, a dog, and a mouse went on an adventure. They didn't know what lay ahead! Surprises awaited them at every turn, and the path was full of wonder. A cat, a dog, and a mouse went on an adventure. They didn't know what lay ahead! Surprises awaited them at every turn, and the path was full of wonder. A cat, a dog, and a mouse went on an adventure. They didn't know what lay ahead! Surprises awaited them at every turn, and the path was full of wonder. A cat, a dog, and a mouse went on an adventure. They didn't know what lay ahead! Surprises awaited them at every turn, and the path was full of wonder."
    truncated_text, token_count = truncate_to_desired_tokens(text)
    print(f"Truncated Text ({token_count} tokens):\n{truncated_text}")

