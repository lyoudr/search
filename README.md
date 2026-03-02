# Search (Audio + Medical LLM + Agent Workflow)

`search` is a FastAPI backend for:
- audio preprocessing and Whisper transcription
- medical text correction (direct LLM / RAG / agent-based)
- vector retrieval with Pinecone
- WER evaluation and statistics

## Tech Stack

- FastAPI
- SQLAlchemy + Alembic
- PostgreSQL
- OpenAI / Hugging Face model integration
- Pinecone (vector DB)
- Whisper (ASR)
- AWS Transcribe / Google Speech-to-Text

## Current Architecture
![Architecture](https://github.com/lyoudr/search/blob/feature/ai-agent/architecture.png)

### Routers

- `app/routes/audio.py`
  - audio parsing/splitting utilities
  - XLSX export
- `app/routes/llm.py`
  - batch correction (`/llm`, `/llm/hematology`, `/llm/agent`)
  - whisper analyze (`/llm/whisper/analyze`)
  - Google/AWS batch transcription
- `app/routes/wer.py`
  - whisper/llm WER calculation
  - model comparison and statistics
- `app/routes/documents.py`
  - document upload/process/search
  - hematology dictionary ingestion
- `app/routes/models.py`
  - model registry/load/download endpoints

### Agents

- `app/ai_agents/agents/whisper_agent.py`
- `app/ai_agents/agents/llm_agent.py`
- `app/ai_agents/agents/transcription_agent.py`
- `app/ai_agents/agents/evaluation_agent.py`

### Tools

- `app/ai_agents/tools/whisper_tools.py`
- `app/ai_agents/tools/correction_tools.py`
- `app/ai_agents/tools/agent_tools.py`
- `app/ai_agents/tools/shared_vector_tools.py`

### Core Services

- `app/services/audio_core.py`
- `app/services/correction_core.py`
- `app/services/quality_evaluator.py`
- `app/services/document_services.py`
- `app/services/hematology_services.py`
- `app/services/medical_documents_services.py`

## Main Workflows

### 1) Whisper Analyze + Vectorization

`POST /llm/whisper/analyze`

1. `whisper_to_text()` transcribes audio with Whisper
2. save to `audio_files` / `transcriptions`
3. `WhisperAgent` runs:
   - `TermExtractorTool` -> `EmbeddingTool` -> `PineconeUpsertTool` (`query-index`, medical terms)
   - `ChunkTool` -> `EmbeddingTool` -> `PineconeUpsertTool` (`query-index`, transcription chunks)

### 2) LLM Correction

- `POST /llm` (direct or medical-documents RAG)
- `POST /llm/hematology` (hematology RAG)
- `POST /llm/agent` (TranscriptionAgent dynamic strategy)

`LLMAgent` dispatches correction tools for non-agent batch modes.  
Shared correction logic is centralized in `app/services/correction_core.py`.

### 3) WER Evaluation

- `POST /wer/whisper`
- `POST /wer/llm`
- `GET /wer/llm/model-comparison`
- `POST /wer/statistics/recalculate`

## Project Structure (Key Paths)

- `app/main.py` - FastAPI app entry
- `app/models/` - ORM models
- `app/repositories/` - DB access layer
- `app/routes/` - API routes
- `app/services/` - business logic
- `app/ai_agents/` - agent and tool system
- `alembic/` - migrations

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` in project root:

```env
DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>:<port>/<db>
SOURCE_DIR=<absolute path to app/sources>
OPENAI_API_KEY=<your_openai_key>
PINECONE_API_KEY=<your_pinecone_key>
HF_TOKEN=<optional_hf_token>
AWS_REGION=us-east-1
AWS_S3_BUCKET=<optional_for_aws_transcribe>
```

### 3. Run DB migrations

```bash
alembic upgrade head
```

### 4. Start API

```bash
uvicorn app.main:app --reload
```

Default docs:
- `http://127.0.0.1:8000/docs`

## Useful Endpoints (Examples)

- `POST /audio/parse`
- `POST /audio/split`
- `POST /llm/whisper/analyze`
- `POST /llm`
- `POST /llm/hematology`
- `POST /llm/agent`
- `POST /wer/llm`
- `GET /wer/llm/model-comparison`
- `POST /documents/upload`
- `POST /documents/search`

## Notes

- Default LLM in current routes/services is `gpt-5.2` (internally mapped by model manager config).
- `query-index` stores both extracted medical terms and transcription chunks.
- For detailed methodology and workflow diagrams, see `METHODOLOGY.md`.