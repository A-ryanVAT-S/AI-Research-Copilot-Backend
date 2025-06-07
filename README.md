# 🤖 AI Research Copilot - Backend

A **FastAPI-based intelligent backend** that processes research papers with **instant AI capabilities**. Upload a PDF and get immediate access to summarization, translation, and Q&A functionality.

## 🌟 Key Features

### 📄 **Instant Processing**
- **Immediate summarization** upon PDF upload
- **Pre-built Q&A index** for instant question answering
- **Zero waiting time** for core functionality

### 🤖 **AI-Powered Capabilities**
- **Intelligent Summarization**: BART-Large CNN model for research papers
- **Multi-language Translation**: Helsinki-NLP models (11+ languages)
- **Smart Q&A System**: Mistral 7B with FAISS vector search
- **Document Understanding**: Semantic chunking with overlap

### ⚡ **Performance Optimized**
- **Lazy-loaded models** with LRU caching
- **Batch processing** for translations
- **Memory management** with CUDA optimization
- **Persistent document storage** in memory

---

## 📊 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Extraction │───▶│   AI Processing │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────────────────────────────────────────────┼─────────────────┐
│                    Parallel Processing                  ▼                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  Summarization  │  │  FAISS Indexing │  │  Text Cleaning  │          │
│  │   (BART-CNN)    │  │ (Embeddings)    │  │  (Preprocessing)│          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation & Setup

### 1️⃣ **Environment Setup**
```powershell
# Clone the repository
git clone https://github.com/A-ryanVAT-S/AI-Research-Copilot-Backend.git
cd AI-Research-Copilot-Backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ **PyTorch Configuration**
```powershell
# For CUDA 12.1+ (GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only deployment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3️⃣ **Download Required Models**
```powershell
# Download Mistral 7B GGUF model for Q&A
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf" -OutFile "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
```

### 4️⃣ **Start the Server**
```powershell
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📚 API Endpoints

### **File Management**
- `POST /api/upload` - Upload PDF and process immediately
- `GET /api/file-info/{doc_id}` - Get document metadata

### **AI Features**
- `GET /api/analyze/{doc_id}` - Get document summary
- `POST /api/translate/{doc_id}` - Translate summary to target language
- `POST /api/qa/{doc_id}` - Ask questions about the document

### **Example Request**
```python
import requests

# Upload and process document
files = {'file': open('research_paper.pdf', 'rb')}
response = requests.post('http://localhost:8000/api/upload', files=files)
doc_id = response.json()['doc_id']

# Get instant summary
summary = requests.get(f'http://localhost:8000/api/analyze/{doc_id}')

# Ask questions
qa_response = requests.post(f'http://localhost:8000/api/qa/{doc_id}', 
                           json={'question': 'What is the main contribution?'})
```

---

## 🏗️ Project Structure

```
AI-Research-Copilot-Backend/
├── main.py                 # FastAPI application & endpoints
├── summaryModel.py         # BART-based summarization
├── translationModel.py     # Helsinki-NLP translation models
├── qaModel.py             # Mistral 7B Q&A with FAISS
├── requirements.txt       # Python dependencies
├── files/                 # Uploaded documents storage
└── __pycache__/          # Python bytecode cache
```

---

## 🔧 Configuration

### **Environment Variables**
```bash
# Optional configuration
TORCH_DEVICE=cuda          # Force device selection
MODEL_CACHE_DIR=./models   # Custom model cache directory
MAX_FILE_SIZE=10485760     # File size limit (10MB)
```

### **Hardware Requirements**
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU, NVIDIA GPU
- **Storage**: 10GB for models and cache
- **Python**: 3.9+ (tested with 3.12)

---

## 🚀 Performance Features

### **Memory Optimization**
- LRU cache for models (max 5 translation models)
- Automatic CUDA memory cleanup
- Batch processing for large documents

### **Processing Speed**
- **Upload to Summary**: ~30-60 seconds
- **Q&A Response**: ~5-10 seconds
- **Translation**: ~3-8 seconds

### **Scalability**
- Stateless design for horizontal scaling
- In-memory document store (can be replaced with Redis/DB)
- Lazy model loading reduces startup time

---

## 🛡️ Security & Limitations

### **Security Features**
- File type validation (PDF/text only)
- File size limits (10MB max)
- Input sanitization and error handling

### **Current Limitations**
- **Memory Storage**: Documents stored in RAM (not persistent)
- **Single Instance**: No multi-user session management
- **Model Size**: Large models require significant RAM
- **Language Support**: 11 languages for translation

---

## 🔄 Future Enhancements
- [ ] Database integration for persistent storage
- [ ] Redis caching for production deployment
- [ ] User authentication and session management
- [ ] Batch processing for multiple documents
- [ ] Additional model options and fine-tuning
- [ ] Real-time WebSocket updates for processing status

---

## 📞 Support & Contributing

**Issues**: Report bugs and feature requests on GitHub  
**Documentation**: See `/docs` for detailed API documentation  
**Contributing**: Fork, branch, and submit pull requests  

**Tech Stack**: FastAPI, PyTorch, Transformers, LangChain, FAISS

