from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# Global variables for storing context and model
index, model, texts = None, None, []


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_faiss_index(texts):
    global index, model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    # Create FAISS index
    d = embeddings.shape[1]  # dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global texts

    # Extract text from PDF
    pdf_content = extract_text_from_pdf(file.file)
    
    # Split text into paragraphs or segments
    paragraphs = pdf_content.split('\n\n')
    for para in paragraphs:
        if len(para) > 3000:
            # break at 1000 characters and keep all 1000 characters
            for i in range(0, len(para), 3000):
                texts.append(para[i:i+3000])
        else:
            texts.append(para)
            
    

    # Rebuild FAISS index with updated texts
    create_faiss_index(texts)
    
    return {"message": "PDF content has been uploaded and added to context."}


@app.delete("/delete_context")
async def delete_context():
    global texts, index

    # Clear the context and reset index
    texts = []
    index = None
    return {"message": "All context has been deleted."}


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_context(request: QueryRequest):
    global index, model, texts
    
    if not index or not texts:
        raise HTTPException(status_code=400, detail="No context is available to query.")
    
    # Encode the query
    query_embedding = model.encode([request.query])
    
    # Search in the FAISS index
    distances, indices = index.search(query_embedding, k=3)
    
    # Fetch the results from the text data
    results = [texts[idx] for idx in indices[0]]
    
    return {"query": request.query, "results": results}


@app.on_event("startup")
async def startup_event():
    global model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
