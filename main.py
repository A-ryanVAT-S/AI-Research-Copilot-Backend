import os
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import torch

# Import model components
from translationModel import translate_text_core
from summaryModel import summarize_text_optimized
from qaModel import create_faiss_index, initialize_qa_system, load_pdf_text

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Add CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this line
)

# Document storage and processing state
document_store: Dict[str, Dict] = {}


class QAResponse(BaseModel):
    question: str


class TranslationRequest(BaseModel):
    target_lang: str


@app.post("/api/upload")
async def upload_file(file: UploadFile):
    try:
        # Single validation check
        allowed_types = ["application/pdf", "text/plain"]
        if file.content_type not in allowed_types:
            return JSONResponse(
                {"detail": "Only PDF/text files allowed"},
                status_code=400
            )

        # Generate unique IDs and paths
        file_uuid = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        stored_filename = f"{file_uuid}{file_ext}"
        file_path = UPLOAD_DIR / stored_filename

        # Save file to persistent storage
        content = await file.read()

        # File size validation (10MB limit)
        if len(content) > 10 * 1024 * 1024:
            return JSONResponse(
                {"detail": "File size exceeds 10MB limit"},
                status_code=400
            )

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Extract and store text
        text = load_pdf_text(str(file_path))
        if not text.strip():
            raise ValueError("Failed to extract text from PDF")

        # # Delete the file after processing
        # try:
        #     os.remove(file_path)
        # except Exception as e:
        #     print(f"Error deleting file: {str(e)}")

        # Store document metadata
        document_store[file_uuid] = {
            "text": text,
            "filename": file.filename,
            "faiss_index": None,
            "translations": {},
            "created_at": datetime.now().isoformat()
        }

        return JSONResponse({
            "doc_id": file_uuid,
            "filename": file.filename
        })

    except Exception as e:
        return JSONResponse(
            {"detail": f"Upload failed: {str(e)}"},
            status_code=500
        )


@app.get("/api/file-info/{doc_id}")
def get_file_info(doc_id: str):
    """Get document metadata"""
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    return JSONResponse({
        "doc_id": doc_id,
        "filename": doc["filename"],
        "upload_date": doc["created_at"]
    })


@app.get("/api/analyze/{doc_id}")
def get_summary(doc_id: str):
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        # Check if summary exists
        if "summary" not in doc:
            # Generate and store summary if missing
            doc["summary"] = summarize_text_optimized(doc["text"])

        return JSONResponse({
            "status": "completed",
            "summary": doc["summary"]
        })
    except Exception as e:
        raise HTTPException(500, f"Summarization failed: {str(e)}")




@app.post("/api/translate/{doc_id}")
def get_translation(doc_id: str, request: TranslationRequest = Body(...)):
    """Handle translation requests"""
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        if request.target_lang not in doc["translations"]:
            try:
                translated_text = translate_text_core(
                    doc["text"],
                    request.target_lang
                )
                doc["translations"][request.target_lang] = translated_text
            except ValueError as e:
                raise HTTPException(400, detail=str(e))
            except RuntimeError as e:
                raise HTTPException(500, detail=str(e))

        return PlainTextResponse(
            content=doc["translations"][request.target_lang],
            headers={
                "Content-Disposition":
                    f"attachment; filename={doc['filename']}_{request.target_lang}_translation.txt"
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {str(e)}")


@app.post("/api/qa/{doc_id}")
def answer_question(doc_id: str, request: QAResponse = Body(...)):
    """Handle QA requests"""
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        # Lazy-load FAISS index
        if not doc["faiss_index"]:
            doc["faiss_index"] = create_faiss_index(doc["text"])

        qa = initialize_qa_system(doc["faiss_index"])

        with torch.inference_mode():
            result = qa({"query": request.question})
            torch.cuda.empty_cache()

        return JSONResponse({
            "question": request.question,
            "answer": result["result"],
            "sources": [d.page_content[:200] + "..." for d in result["source_documents"]]
        })
    except Exception as e:
        raise HTTPException(500, f"QA failed: {str(e)}")

# @app.on_event("startup")
# async def startup_event():
#     """Initialize models on startup"""
#     try:
#         # Warm up models
#         dummy_text = "This is a test document."
#         summarize_text_optimized(dummy_text)
#         translate_text_core(dummy_text, "es")
#
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB")
#     except Exception as e:
#         print(f"Startup error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)