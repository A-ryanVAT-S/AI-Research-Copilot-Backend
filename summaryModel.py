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
MODEL_NAME = "facebook/bart-large-cnn"  # Better for research papers and formal documents
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
    text = re.sub(r'Figure \d+[.:]\s*', '', text)
    text = re.sub(r'Table \d+[.:]\s*', '', text)
    # Remove citations
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\w+\s+et\s+al\.,?\s+\d{4}\)', '', text)
    # Remove URLs and DOIs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'doi:\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # Clean up spacing and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def chunk_text(text, tokenizer, chunk_size=900):
    """Create meaningful chunks preserving sentence boundaries"""
    # Split by sentences more intelligently
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Check if adding this sentence would exceed the limit
        test_chunk = ' '.join(current_chunk + [sentence])
        if len(tokenizer.tokenize(test_chunk)) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter very short chunks


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
        
        # Add a prompt prefix to improve summarization quality
        prompted_batch = [f"Summarize this research paper section: {chunk}" for chunk in batch]
        
        inputs = tokenizer(
            prompted_batch,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding='longest',
            return_tensors="pt"
        ).to(DEVICE)
        
        summary_ids = model.generate(
            **inputs,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            max_length=250,
            min_length=80,
            no_repeat_ngram_size=3,
            do_sample=False,
            temperature=1.0
        )

        summaries.extend(tokenizer.batch_decode(summary_ids, skip_special_tokens=True))

    # Combine summaries intelligently
    combined_summary = ' '.join(summaries)
    
    # If combined summary is still too long, create a final summary
    if len(tokenizer.tokenize(combined_summary)) > 400:
        # Create a final summary without recursion
        final_prompt = f"Create a comprehensive summary of this research paper: {combined_summary}"
        final_inputs = tokenizer(
            final_prompt,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        final_summary_ids = model.generate(
            **final_inputs,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            max_length=200,
            min_length=60,
            no_repeat_ngram_size=3,
            do_sample=False
        )
        
        combined_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)

    return combined_summary


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

