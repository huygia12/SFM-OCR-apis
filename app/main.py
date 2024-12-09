from fastapi import APIRouter, FastAPI
from starlette.middleware.cors import CORSMiddleware

from .api import health, ocr
from .core import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

api_router = APIRouter()
api_router.include_router(health.router, tags=['Health check'])
api_router.include_router(ocr.router, prefix=settings.API_V1_STR, tags=['OCR'])
app.include_router(api_router) 
