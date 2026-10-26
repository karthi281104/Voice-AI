# VoiceDesk AI — Browser Voice MVP

This is the first independently testable prototype. It lets a business configure its profile, services, doctors, opening hours and FAQs, then test a knowledge-grounded receptionist through text or browser voice.

## What is included
- FastAPI backend
- SQLite database (created automatically)
- Business configuration form
- LLM-powered text chat using a local Ollama model, grounded in saved business records
- Deterministic fallback if Ollama is not running
- Browser microphone input via Web Speech API where supported
- Browser text-to-speech using installed system/browser voices
- API tests

## Deliberate limitations
- The LLM runs locally through Ollama, so there is no per-request hosted LLM API charge. Model downloads use disk space and local CPU/GPU/RAM.
- If Ollama is unavailable, the app falls back to simple deterministic matching; check the server terminal and Ollama status if answers seem limited.
- Browser speech recognition is browser-dependent; use a recent Chrome or Edge browser and allow microphone access.
- Speech synthesis uses voices available on the user's device. Voice quality varies by OS/browser and installed language voices.
- No phone/telephony integration, authentication, multi-tenancy, appointments that modify a calendar, or WhatsApp yet.
- This is a local prototype, not production-ready for real customer/patient data.

## Requirements
- Python 3.10+
- Recent Chrome or Edge recommended for microphone testing

## Set up the local LLM (one-time)
1. Install Ollama for Windows from https://ollama.com/download.
2. Open a new terminal after installation and download a model:
   ```powershell
   ollama pull qwen3:4b
   ```
3. Confirm Ollama responds:
   ```powershell
   ollama run qwen3:4b
   ```
   Type a test message, then exit the model session. Ollama normally stays available in the background.

The default model is `qwen3:4b`. You can change it with the `OLLAMA_MODEL` environment variable, for example `qwen3:8b` if your computer has enough memory. The first answer may take longer while the model loads.

## Run on Windows (PowerShell)
```powershell
cd voicedesk-ai-mvp
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run on Linux/macOS
```bash
cd voicedesk-ai-mvp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Run tests
```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

## LLM configuration
- `OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`.
- `OLLAMA_MODEL` defaults to `qwen3:4b`.
- `OLLAMA_TIMEOUT_SECONDS` defaults to `45`.

## API
- `GET /api/health`
- `GET /api/business`
- `PUT /api/business`
- `POST /api/chat` with JSON `{"message":"How much is teeth cleaning?","language":"en-IN"}`

## How to enter business information
Use one item per line in the configuration fields:
- Services: `Service name | price | duration | description`
- Doctors: `Doctor name | specialty | availability`
- FAQs: `Question | answer`

Example:
```text
Teeth cleaning | ₹1500 | 30 minutes | Cleaning and polishing
Dr. Priya | Dentist | Mon-Sat, 9 AM-1 PM
Do you accept walk-ins? | Please call the clinic before visiting.
```

## Safety and correctness
- The chat can only answer from configured fields and FAQs.
- If a question does not match the available data, it says it cannot confirm the answer.
- Availability text is descriptive only; this prototype does not check live calendar slots or create appointments.
- Do not enter real patient records, credentials, or sensitive personal data.
