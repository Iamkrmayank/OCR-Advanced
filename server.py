from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile, subprocess, os
import fitz  # PyMuPDF for page counting

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY", "")

# --- CORS ---
_ALLOWED = os.getenv("ALLOWED_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in _ALLOWED.replace(";", ",").split(",") if o.strip()] or ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,  # keep False if you use "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth check ---
def require_key(x_api_key: str | None):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(401, "Unauthorized")

# --- OCR Runner ---
def run_processor(pdf_path: str, title: str | None) -> str:
    """
    Call mistral_ocr-v2.py to generate markdown and DOCX output.
    Locate the resulting DOCX file (script writes under current working directory).
    """
    base_out = "out"  # bare name avoids Path.with_name issues
    script_path = Path(__file__).parent / "mistral_ocr-v2.py"
    cwd = Path(__file__).parent

    cmd = [
        "python", str(script_path), pdf_path,
        "--out", base_out,
    ]
    if title:
        cmd += ["--title", title]

    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    print("=== CMD ===", " ".join(cmd))
    print("=== STDOUT ===\n", p.stdout)
    print("=== STDERR ===\n", p.stderr)

    if p.returncode != 0:
        raise HTTPException(500, f"OCR script failed:\n{p.stderr[:1000]}")

    # Expected outputs to check (first existing wins) - looking for .docx file
    candidates = [
        cwd / f"{base_out}.docx",
        Path(pdf_path).parent / f"{base_out}.docx",
        Path(pdf_path).parent / f"{Path(pdf_path).stem}.docx",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    raise HTTPException(500, "DOCX file not found after processing")

# --- Health check ---
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "has_api_key": bool(API_KEY),
        "has_mistral_key": bool(os.getenv("MISTRAL_API_KEY")),
        "has_mistral_endpoint": bool(os.getenv("MISTRAL_OCR_ENDPOINT")),
    }

# --- Upload OCR (advanced) ---
@app.post("/ocr-advanced")
async def ocr_file(
    file: UploadFile = File(...),
    title: str = Form("My Document"),
    x_api_key: str | None = Header(None)
):
    require_key(x_api_key)

    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(await file.read())
        pdf_path = f.name

    # Credits = number of pages
    try:
        doc = fitz.open(pdf_path)
        credits_used = doc.page_count
        doc.close()
    except Exception as e:
        raise HTTPException(500, f"Failed to count pages: {e}")

    # Run OCR processor (returns DOCX file path)
    docx_path = run_processor(pdf_path, title)

    # Return DOCX file with Content-Length (prevents chunked issues behind Front Door)
    return FileResponse(
        path=docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="output.docx",
        headers={"X-Credits-Used": str(credits_used)},
    )
