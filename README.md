# Project 05 — KYC Copilot

A Streamlit prototype for Intelligent KYC & Customer Onboarding. It demonstrates document intake, field extraction, cross-document checks, attention flags, a draft KYC summary, human review actions, and a session audit log.

## Features

- Upload PDF, TXT, CSV, Markdown, DOCX, or image files.
- Optional Gemini-powered extraction and summary generation.
- Regex/OCR fallback when no Gemini API key is configured.
- Cross-document comparison and missing-document/field checks.
- Five synthetic demo cases.
- Human-in-the-loop review choices and downloadable audit log.
- Evaluation checklist and architecture overview.

## Run locally

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

## Run on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Deploy. The app can run without Gemini by using the fallback mode.

### Optional Gemini API key

In Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

```toml
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

Do not commit API keys to GitHub. If using Colab, set the key as an environment variable or use Colab Secrets.

## OCR notes

- `pytesseract` is a Python wrapper; the Tesseract OCR executable must also be installed for image OCR.
- On Debian/Ubuntu/Colab, install it with:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-vie
```

- Streamlit Community Cloud may require additional system setup for Tesseract. If OCR is unavailable, the app may still use Gemini vision for image files when a key is configured, or work with text-based PDFs and text files.
- Scanned PDFs may need OCR separately; the current PDF path extracts embedded text only.

## Privacy and limitations

This is an academic prototype, not a production KYC system. Use synthetic or anonymized data. Do not upload real identity documents or commit them to the repository. The system does not authenticate government IDs, access government databases, or make final onboarding decisions. Human review is required.
