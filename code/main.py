from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from implementation import PromptAnalysis  # Assuming your class is saved as prompt_analysis.py

app = FastAPI()
analysis = PromptAnalysis()

class PromptRequest(BaseModel):
    prompt: str

class QARequest(BaseModel):
    prompt: str
    response: str

@app.post("/generate-response")
def generate_response(request: PromptRequest):
    try:
        result = analysis.generate_response(request.prompt)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-confidence")
def get_confidence(request: PromptRequest):
    try:
        confidence = analysis.get_confidence(request.prompt)
        return {"confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-entropy")
def get_entropy(request: PromptRequest):
    try:
        entropy = analysis.get_entropy(request.prompt)
        return {"entropy": entropy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa-reversibility")
def qa_reversibility(request: QARequest):
    try:
        similarity = analysis.qa_reversibility(request.prompt, request.response)
        return {"similarity": similarity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
