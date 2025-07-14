import numpy as np
import re

def clean_text(text: str) -> str:
    text = text.lower()
    # Remove all chinese and English 標點
    text = re.sub(r"[，。！？：；「」『』、（）《》〈〉【】〔〕…—．·,\.!?;:'\"()\[\]{}]", " ", text)
    tokens = []
    buffer = ""

    for char in text: 
        if re.match(r"[a-z0-9%]",char):
            buffer += char 
        else:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            if char.strip():
                tokens.append(char)

    if buffer:
        tokens.append(buffer)
    return tokens

def wer(reference: str, hypothesis: str) -> float:
    ref_words = clean_text(reference)
    hyp_words = clean_text(hypothesis)
    r_len = len(ref_words)
    h_len = len(hyp_words)

    # Initialize the matrix
    d = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j
    print("d is ->", d)
    
    
    # Compute edit distance
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # Deletion
                d[i][j - 1] + 1,        # Insertion
                d[i - 1][j - 1] + cost  # Substitution
            )

    return round(d[r_len][h_len] / r_len, 2) if r_len > 0 else 0.0