from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import fitz  # PyMuPDF
import re
from functools import lru_cache

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32
BATCH_SIZE = 2  # Reduced for better memory management
MODEL_NAME = "philschmid/bart-large-cnn-samsum"  # Better for formal documents
MAX_INPUT_LENGTH = 1024


@lru_cache(maxsize=1)
def load_resources():
    """Load and cache summarization model with progress tracking"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE
    ).to(DEVICE)
    if DEVICE == 'cuda':
        print(f"✅ Model loaded on {torch.cuda.get_device_name(0)}", flush=True)
    return tokenizer, model


def clean_text(text):
    """Preprocess research paper text"""
    # Remove headers/footers
    text = re.sub(r'Page \d+ of \d+', '', text)
    # Remove citations
    text = re.sub(r'\[\d+\]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove multiple newlines
    return re.sub(r'\n+', '\n', text).strip()


def chunk_text(text, tokenizer, chunk_size=800):
    """Create meaningful chunks preserving sentence boundaries"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(tokenizer.tokenize(' '.join(current_chunk + [sentence]))) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def summarize_text_optimized(text):
    """Research paper optimized summarization"""
    tokenizer, model = load_resources()

    # Clean and chunk text
    cleaned_text = clean_text(text)
    chunks = chunk_text(cleaned_text, tokenizer)


    # Process chunks with overlap
    summaries = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding='longest',
            return_tensors="pt"
        ).to(DEVICE)

        summary_ids = model.generate(
            **inputs,
            num_beams=6,
            repetition_penalty=3.0,
            length_penalty=2.5,
            early_stopping=True,
            max_length=300,
            min_length=100,
            no_repeat_ngram_size=3
        )

        summaries.extend(tokenizer.batch_decode(summary_ids, skip_special_tokens=True))

    # Final refinement
    final_summary = ' '.join(summaries)
    if len(tokenizer.tokenize(final_summary)) > 500:
        final_summary = summarize_text_optimized(final_summary)

    return final_summary


def summarize_document(input_path):
    """Academic paper focused summarization"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"🚫 File not found: {input_path}")

    print(f"📄 Processing research paper: {os.path.basename(input_path)}", flush=True)

    # Improved PDF text extraction
    with fitz.open(input_path) as doc:
        text = []
        for page in doc:
            text.append(page.get_text("text", flags=fitz.TEXT_PRESERVE_IMAGES))
        full_text = '\n'.join(text)

    if not full_text.strip():
        raise ValueError("📄 Document contains no readable text")

    return summarize_text_optimized(full_text)

