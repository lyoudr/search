import re 
from typing import List 
from jiwer import wer as jiwer_wer

def clean_text(text: str) -> str:
    """
    Tokenize text for WER calculation.

    - English: word-level tokens (a-z, 0-9, %)
    - Chinese: character-level tokens
    - Removes all punctuation
    """
    text = text.lower() 
    # Replace Chinese and English punctuation with spaces
    text = re.sub( 
        r"[，。！？：；「」『』、（）《》〈〉【】〔〕…—．·,\.!?;:'\"()\[\]{}]",
        " ",
        text
    )
    tokens: List[str] = []
    buffer = ""

    for char in text: 
        if re.match(r"[a-z0-9%]", char):
            buffer += char 
        else:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            if char.strip():  # keep non-whitespace (Chinese)
                tokens.append(char)
    
    if buffer:
        tokens.append(buffer)
    return tokens


def wer(reference: str, hypothesis: str) -> float: 
    """
    Compute Word Error Rate (WER) using jiwer with a custom tokenizer.

    Returns a float between 0 and 1.
    """
    ref_tokens = clean_text(reference)
    hyp_tokens = clean_text(hypothesis)

    # Join tokens back to string for jiwer
    ref_str = " ".join(ref_tokens) if ref_tokens else ""
    hyp_str = " ".join(hyp_tokens) if hyp_tokens else ""

    # Use jiwer to compute WER 
    return round(jiwer_wer(ref_str, hyp_str), 2)