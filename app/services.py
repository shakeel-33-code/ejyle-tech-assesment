import os
import shutil
import uuid
import docx
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader
from transformers import pipeline

# In-memory storage for file metadata (in a real app, use a database)
file_storage = {}

# Create a directory to store uploaded files
UPLOAD_FOLDER = "uploaded_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the AI models only once when the server starts
try:
    print("Loading AI models. This may take a moment...")
    summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    print("AI models loaded successfully.")
except Exception as e:
    print(f"Error loading AI models: {e}")
    # In a production app, you might want to exit here or log the error
    summarizer_model = None
    qa_model = None

def extract_text(file_path: str):
    """Extracts text from a given file path based on its extension."""
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif file_extension.lower() == ".pdf":
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading PDF file: {e}")

    elif file_extension.lower() == ".docx":
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading DOCX file: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

def upload_file_service(file: UploadFile):
    """Handles the logic for uploading and saving a file."""
    valid_extensions = {".txt", ".pdf", ".docx"}
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension.lower() not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .txt, .pdf, or .docx file."
        )

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, file_id + file_extension)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    file_storage[file_id] = {
        "filename": file.filename,
        "path": file_path,
        "size": file.size
    }

    return {"message": "File uploaded successfully", "file_id": file_id}

def summarize_document_service(file_id: str):
    """Handles the summarization logic."""
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_storage[file_id]
    
    try:
        document_text = extract_text(file_info["path"])

        if not document_text.strip():
            raise HTTPException(status_code=400, detail="Document is empty or no text could be extracted.")
        
        # Truncate text for summarization model, as it has an input token limit
        max_length = 1500
        if len(document_text) > max_length:
            document_text = document_text[:max_length]
        
        summary = summarizer_model(document_text, max_length=150, min_length=30, do_sample=False)
        return {"file_id": file_id, "summary": summary[0]["summary_text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during summarization: {e}")

def ask_question_service(file_id: str, question: str):
    """Handles the question-answering logic."""
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_storage[file_id]
    
    try:
        context = extract_text(file_info["path"])

        if not context.strip():
            raise HTTPException(status_code=400, detail="Document is empty or no text could be extracted.")
        
        answer = qa_model(question=question, context=context)

        return {
            "file_id": file_id,
            "question": question,
            "answer": answer["answer"],
            "score": answer["score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while answering the question: {e}")