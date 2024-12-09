from fastapi import APIRouter, UploadFile, File, HTTPException
from app.service import do_ocr
from pydantic import BaseModel
from typing import List

router = APIRouter()

class OCRRequest(BaseModel):
    image_urls: List[str]
    application_name: str

@router.post("/")
async def perform_ocr(request: OCRRequest):
    image_urls = request.image_urls
    application_name = request.application_name
    
    text = await do_ocr(image_urls, application_name)
    return {"extracted_text": text}