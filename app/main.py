
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables before telemetry initialization.
load_dotenv(override=True)

from microsoft.opentelemetry import use_microsoft_opentelemetry

# Initialize Azure Application Insights telemetry.
use_microsoft_opentelemetry(
    enable_azure_monitor=True,
    logger_name="piassist",
)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.services.audit import init_db
from app.api.routes import router

configure_logging()
settings = get_settings()
init_db()

logger = logging.getLogger("piassist")
logger.setLevel(logging.INFO)

app = FastAPI(title=settings.app_name, version="1.0.0")

app.include_router(router)
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logger.info("Homepage requested")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": settings.app_name},
    )
