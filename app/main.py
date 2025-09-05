from fastapi import FastAPI, UploadFile, File, HTTPException
from . import services
from .models import Question

# Initialize FastAPI app
app = FastAPI(
    title="Document AI Service",
    description="API for uploading, summarizing, and questioning documents.",
)

# 1. Upload and Store File
@app.post("/uploadfile/", tags=["File Management"])
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a document and stores it with a unique ID.
    Supported file types: .txt, .pdf, .docx
    """
    try:
        return services.upload_file_service(file)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# 2. Summarize Document
@app.get("/summarize/{file_id}", tags=["AI Operations"])
def summarize_document(file_id: str):
    """
    Generates a concise AI-based summary of the document content.
    """
    try:
        return services.summarize_document_service(file_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# 3. Question Answering on Document
@app.post("/ask/{file_id}", tags=["AI Operations"])
def ask_question(file_id: str, payload: Question):
    """
    Uses AI to answer a question based on the document's content.
    """
    try:
        return services.ask_question_service(file_id, payload.question)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")