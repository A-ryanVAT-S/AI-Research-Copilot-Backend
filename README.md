# AI Research Copilot - Backend

## Overview
The AI Research Copilot backend is a **FastAPI-powered** system that provides research paper analysis using **state-of-the-art NLP models**. It supports **question-answering, summarization, translation, and document processing** with efficient **vector search** and **transformer-based models**.

## Features
- **FastAPI-based REST API**
- **Hugging Face Transformers** for NLP tasks
- **FAISS for efficient vector search**(currently issue in faiss versions)
- **PDF processing** with PyMuPDF and PyPDF
- **GGUF model support** for lightweight inference
- **Support for CPU and GPU (CUDA)**

---

## Installation

### 1. **Clone the Repository**
```sh
git clone https://github.com/A-ryanVAT-S/AI-Research-Copilot.git
cd AI-Research-Copilot
```

### 2. **Create and Activate a Virtual Environment**
```sh
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. **Install Dependencies**

#### Core Dependencies:
```sh
pip install fastapi uvicorn python-multipart pydantic
```

#### Machine Learning Frameworks:
**(Choose one based on your setup)**

**For GPU (CUDA 12.1+):**
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only:**
```sh
pip install torch torchvision torchaudio
```

#### NLP & AI Libraries:
```sh
pip install transformers sentence-transformers ctransformers
pip install langchain langchain-community langchain-core langchain-text-splitters
pip install faiss-cpu
```

#### Document Processing:
```sh
pip install pypdf pymupdf python-dotenv
```

#### Additional Utilities:
```sh
pip install tqdm requests numpy typing-extensions
```

---

## Running the Backend

### **1. Download Required Models**
```sh
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```
(This is required for the QA system.)

### **2. Start the Server**
```sh
uvicorn main:app --reload
```

---

## System Requirements
- **Python**: 3.9+
- **RAM**: 8GB+ (16GB recommended)
- **Disk Space**: 10GB+ (for models and dependencies)
- **GPU (Optional)**: NVIDIA GPU with CUDA 12.1+

---

## Notes
- The first run will **download necessary Hugging Face models**, which may take time.
- Ensure your **Python version is 3.9 or higher** to avoid compatibility issues.
- If using a **GPU**, install the correct **CUDA-compatible PyTorch version**.

---

