# Mistral OCR API

A FastAPI-based OCR service that converts PDF documents to Markdown and DOCX formats using Mistral AI's Document AI API. The service provides both REST API endpoints and command-line tools for document processing.

## Features

- 📄 **PDF to Markdown**: Convert PDF documents to clean Markdown format
- 📝 **PDF to DOCX**: Generate well-formatted Word documents with proper tables, lists, and formatting
- 🔒 **API Key Authentication**: Secure API access with custom API keys
- 🐳 **Docker Support**: Easy deployment with Docker
- 📊 **Table Support**: Properly formatted tables in DOCX output (v3, v4)
- 📋 **List Support**: Bulleted and numbered lists support (v4)
- 🎨 **Better Formatting**: Improved document styling with margins, spacing, and alignment (v4)

## Project Structure

```
mistral-ocr-api/
├── server.py              # FastAPI server
├── mistral_ocr.py         # Original OCR script
├── mistral_ocr-v2.py      # Simple version (text-only, no images)
├── mistral_ocr-v3.py      # With table parsing support
├── mistral_ocr-v4.py      # Enhanced version with better formatting
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── .env                  # Environment variables (create this)
```

## Installation

### Prerequisites

- Python 3.11+
- Mistral AI API credentials

### Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd mistral-ocr-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```env
MISTRAL_OCR_ENDPOINT=https://your-endpoint.services.ai.azure.com/providers/mistral/azure/ocr
API_KEY=your_api_key_here
MISTRAL_MODEL=mistral-document-ai-2505
```

4. Run the server:
```bash
uvicorn server:app --host 0.0.0.0 --port 8010 --reload
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `MISTRAL_OCR_ENDPOINT` | Mistral OCR API endpoint URL | Yes |
| `API_KEY` | Your Mistral API key | Yes |
| `MISTRAL_MODEL` | Model name (default: `mistral-document-ai-2505`) | No |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated, default: `*`) | No |

## Usage

### Command Line

#### Basic Usage
```bash
python mistral_ocr-v4.py "document.pdf" --out output
```

#### With Title
```bash
python mistral_ocr-v4.py "document.pdf" --out output --title "My Document"
```

#### Without Page Breaks
```bash
python mistral_ocr-v4.py "document.pdf" --out output --no-page-breaks
```

**Output**: Generates both `output.md` (Markdown) and `output.docx` (Word document)

### API Endpoints

#### Health Check
```bash
GET /healthz
```

**Response**:
```json
{
  "ok": true,
  "has_api_key": true,
  "has_mistral_key": true,
  "has_mistral_endpoint": true
}
```

#### OCR Processing
```bash
POST /ocr-advanced
```

**Headers**:
```
x-api-key: your_api_key
Content-Type: multipart/form-data
```

**Form Data**:
- `file`: PDF file (required)
- `title`: Document title (optional, default: "My Document")

**Response**: Returns DOCX file with headers:
- `Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- `X-Credits-Used: <number_of_pages>`

**Example with curl**:
```bash
curl -X POST "http://localhost:8010/ocr-advanced" \
  -H "x-api-key: your_api_key" \
  -F "file=@document.pdf" \
  -F "title=My Document" \
  --output output.docx
```

**Example with PowerShell**:
```powershell
$filePath = "document.pdf"
$apiKey = "your_api_key"
$uri = "http://localhost:8010/ocr-advanced"

$formData = @{
    file = Get-Item $filePath
    title = "My Document"
}

$headers = @{
    "x-api-key" = $apiKey
}

Invoke-RestMethod -Uri $uri -Method Post -Form $formData -Headers $headers -OutFile "output.docx"
```

## Docker Deployment

### Build Image
```bash
docker build -t mistral-ocr-api .
```

### Run Container
```bash
docker run -d \
  -p 8010:8010 \
  -e MISTRAL_OCR_ENDPOINT="https://your-endpoint..." \
  -e API_KEY="your_api_key" \
  -e MISTRAL_MODEL="mistral-document-ai-2505" \
  mistral-ocr-api
```

### Docker Compose
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  ocr-api:
    build: .
    ports:
      - "8010:8010"
    environment:
      - MISTRAL_OCR_ENDPOINT=${MISTRAL_OCR_ENDPOINT}
      - API_KEY=${API_KEY}
      - MISTRAL_MODEL=${MISTRAL_MODEL}
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Version Differences

### mistral_ocr.py
- Original version with full image support
- Supports Pandoc for math and tables
- Includes image extraction and cropping

### mistral_ocr-v2.py
- Text-only mode (no images)
- Simple table handling (as paragraphs)
- Clean markdown output

### mistral_ocr-v3.py
- Text-only mode
- **Proper table parsing** - converts markdown tables to DOCX tables
- Better table formatting

### mistral_ocr-v4.py (Recommended)
- Text-only mode
- **Proper table parsing** with alignment support
- **List support** (bulleted and numbered)
- **Enhanced formatting**:
  - Better margins and spacing
  - Improved paragraph formatting
  - Header row styling in tables
  - Cell alignment based on markdown

## API Authentication

The API uses API key authentication. Set your API key in the `.env` file:

```env
API_KEY=your_secret_api_key
```

Then include it in requests:
```
x-api-key: your_secret_api_key
```

## Credits System

The API returns the number of pages processed in the `X-Credits-Used` header. Each page counts as one credit.

## Troubleshooting

### Common Issues

1. **"Unauthorized" Error**
   - Check that `API_KEY` is set in `.env`
   - Verify the `x-api-key` header is included in requests

2. **"OCR script failed" Error**
   - Verify `MISTRAL_OCR_ENDPOINT` and `API_KEY` are correct
   - Check Mistral API service status

3. **Tables not formatting correctly**
   - Use `mistral_ocr-v3.py` or `mistral_ocr-v4.py` for proper table support
   - Ensure markdown tables have proper separator rows (`|:--:|`)

4. **Docker build fails**
   - Ensure all required files are present
   - Check Dockerfile paths match your file structure

## Development

### Running Tests
```bash
# Test health endpoint
curl http://localhost:8010/healthz

# Test OCR endpoint
curl -X POST "http://localhost:8010/ocr-advanced" \
  -H "x-api-key: your_key" \
  -F "file=@test.pdf" \
  -F "title=Test" \
  --output test_output.docx
```

### Project Scripts

- `mistral_ocr-v4.py`: Recommended for production (best formatting)
- `mistral_ocr-v3.py`: Good table support
- `mistral_ocr-v2.py`: Simple text-only version
- `mistral_ocr.py`: Full-featured with image support

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

## Support

For issues and questions, please open an issue on GitHub.

