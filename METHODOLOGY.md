# Methods / Methodology

## 3.1 System Architecture

The research system is implemented as a FastAPI-based web application (`app/main.py`) with a modular architecture. The system consists of four main API routers:

- **Audio Processing Router** (`app/routes/audio.py`): Handles audio preprocessing, Whisper transcription, LLM-based text correction, and evaluation metrics computation.
- **Document Processing Router** (`app/routes/documents.py`): Manages document ingestion, chunking, embedding, and storage in vector databases.
- **Model Management Router** (`app/routes/models.py`): Provides a unified model registry supporting multiple LLM providers (OpenAI, HuggingFace, etc.).
- **Generation Router** (`app/routes/generate.py`): Optional endpoints for text generation tasks.

All data persistence is managed through SQLAlchemy ORM models (`app/models/analyze.py`) and repository classes (`app/repositories/*.py`), ensuring a clean separation between business logic and data access.

---

## 3.2 Data Collection and Storage

### 3.2.1 Audio Data Sources

The dataset consists of clinical audio recordings stored in the following directory structure:

- **Raw recordings**: Original `.m4a` files located in `app/sources/origin/`
- **Converted WAV files**: Processed audio files in `app/sources/parse/` (converted from original formats)
- **Split audio chunks**: Segmented `.wav` files in `app/sources/splited/` (prepared for transcription)

### 3.2.2 Database Schema

The system uses a PostgreSQL database with the following core entities (defined in `app/models/analyze.py`):

**AudioFile** (`audio_files` table):
- Stores metadata for each audio file: `id`, `file_path`, `duration_sec`, `language`
- Managed by `audio_file_repository.py`

**Transcription** (`transcriptions` table):
- Stores Whisper transcription results: `id`, `audio_file_id`, `engine`, `text`, `created_at`
- Managed by `transcription_repository.py`
- One-to-many relationship with `AudioFile`

**LLMOutput** (`llm_outputs` table):
- Stores corrected texts from different methods:
  - `text`: Direct LLM correction (baseline, no RAG)
  - `text_with_rag`: Medical document RAG-enhanced correction
  - `text_with_hematology`: Hematology dictionary RAG-enhanced correction
  - `text_agent`: Agent-based hybrid correction
  - `text_with_google`, `text_with_aws`, `text_with_dr_ai`: Cloud service transcriptions
- Managed by `llm_output_repository.py`
- Links to `Transcription` and `LLMModel`

**Evaluation** (`evaluations` table):
- Stores ground truth and WER metrics:
  - `ground_truth`: Reference text for evaluation
  - `whisper_wer`: WER for raw Whisper transcription
  - `llm_wer`: WER for direct LLM correction
  - `llm_rag_wer`: WER for medical document RAG correction
  - `llm_hematology_wer`: WER for hematology dictionary RAG correction
  - `llm_agent_wer`: WER for agent-based correction
- Managed by `evaluation_repository.py`

This schema enables **fair comparison** of all correction methods on the same set of utterances, as each `Transcription` can have multiple `LLMOutput` records (one per method), and each `LLMOutput` can have a corresponding `Evaluation` record.

---

## 3.3 Audio Preprocessing

### 3.3.1 Format Conversion

**Purpose**: Convert heterogeneous audio formats to a uniform `.wav` format compatible with Whisper.

**Implementation**: `POST /audio/parse` endpoint (`app/routes/audio.py:29-40`)

The conversion process is handled by `convert_to_wav()` in `app/services/parse_audio.py`:

1. Recursively scans the input directory (`app/sources/origin/`) for audio files
2. Uses FFmpeg (or equivalent audio processing library) to convert each file to `.wav` format
3. Writes converted files to the output directory (`app/sources/parse/`)

**Replication**: 
- Place original audio files in `app/sources/origin/`
- Call `POST /audio/parse` with `{"input_dir": "origin", "output_dir": "parse"}`

### 3.3.2 Audio Segmentation

**Purpose**: Split long audio recordings into shorter, more manageable chunks for transcription.

**Implementation**: `POST /audio/split` endpoint (`app/routes/audio.py:42-53`)

The segmentation is performed by `split_audio_service()` in `app/services/split_audio.py`:

1. Iterates through all audio files in the input directory
2. Splits each file into fixed-length segments (configurable duration)
3. Stores segmented chunks in the output directory (`app/sources/splited/`)

**Replication**:
- Place `.wav` files in `app/sources/parse/`
- Call `POST /audio/split` with `{"input_dir": "parse", "output_dir": "splited"}`

---

## 3.4 Speech-to-Text Transcription

### 3.4.1 Whisper Transcription

**Purpose**: Transcribe all audio chunks using OpenAI Whisper and store results in the database.

**Implementation**: `POST /audio/whisper/analyze` endpoint (`app/routes/audio.py:56-78`)

The transcription pipeline is implemented in `whisper_to_text()` (`app/services/speech2text.py:29-94`):

**Step 1: Audio File Registration**
```python
for root, _, files in os.walk(input_dir):
    if file_name.endswith((".wav", ".mp3", ".m4a")):
        existing_audio_file = audio_file_repository.get_audio_file_by_path(db, file_path)
        if not existing_audio_file:
            audio_file = audio_file_repository.create_audio_file(
                db=db, file_path=file_path, language="zh"
            )
```

**Step 2: Transcription Execution**
- Uses OpenAI Whisper API (`client.audio.transcriptions.create`) with:
  - Model: `whisper-1`
  - Language: `"zh"` (Chinese)
- Converts simplified Chinese output to traditional Chinese using OpenCC (`OpenCC('s2t')`)

**Step 3: Database Storage**
```python
transcription = transcription_repository.create_transcription(
    db=db,
    audio_file_id=audio_file_id,
    engine=engine,
    text=whisper_text
)
```

This produces the **Whisper baseline transcription** stored in `Transcription.text`, which serves as the input for all subsequent correction methods.

**Replication**:
- Ensure audio files are in `app/sources/splited/`
- Call `POST /audio/whisper/analyze` with `{"input_dir": "splited"}`
- Optionally specify `extraction_model` (default: `"gpt-4o"`) for medical term extraction

### 3.4.2 Automatic Medical Term Extraction

**Purpose**: Extract domain-specific medical terms from each transcription to enable targeted RAG retrieval.

**Implementation**: Integrated into `whisper_to_text()` at `app/services/speech2text.py:84-92`

After creating each `Transcription`, the system automatically calls:

```python
term_processor = TranscriptionTermProcessor()
term_processor.process_transcription(
    db, transcription.id, extraction_model=extraction_model
)
```

**Process** (`app/services/transcription_term_processor.py:26-65`):

1. **Term Extraction**: Uses an LLM (`extraction_model`, default `"gpt-4o"`) to extract key medical terms from `Transcription.text`
2. **Vector Storage**: Embeds extracted terms using `EmbeddingService` and stores them in a Pinecone index (`query-index`) with metadata:
   - `transcription_id`: Links terms to the source transcription
   - `type`: `"medical_term"` (for filtering)
   - `term`: The extracted term text

These stored terms are later used by `MedicalDocumentRetriever` and `HematologyRetriever` as query keywords for semantic search in their respective knowledge bases.

**Replication**: This step is automatic when calling `POST /audio/whisper/analyze`. No separate API call is required.

---

## 3.5 Knowledge Base Construction (RAG)

### 3.5.1 Medical Document Vector Store

**Purpose**: Build a semantic search index of medical literature for general medical terminology RAG.

**Implementation**: 
- `POST /documents/upload` – upload and process new documents
- `POST /documents/process/{file_name}` – process existing documents in the codebase
- Both endpoints in `app/routes/documents.py`

**Core Pipeline** (`DocumentProcessor.process_document` in `app/services/document_processor.py:36-88`):

**Step 1: Document Parsing**
- `parse_document(file_path)` (`app/services/document_parser.py`) extracts plain text from:
  - `.doc` / `.docx` files (using `python-docx` or similar)
  - `.txt` files (direct text reading)

**Step 2: Text Chunking**

Documents are split into overlapping chunks using `TextChunker` (`app/services/text_chunker.py:47-69`):

The chunking process uses a sliding window approach with configurable parameters:

- **Chunk Size**: Maximum characters per chunk (default: `chunk_size = 512`)
- **Chunk Overlap**: Number of overlapping characters between consecutive chunks (default: `chunk_overlap = 50`)

The chunking formula ensures continuity:

```
Chunk_i starts at position: i × (chunk_size - chunk_overlap)
Chunk_i ends at position: i × (chunk_size - chunk_overlap) + chunk_size
```

Where `i` is the chunk index (0-based).

The implementation uses `RecursiveCharacterTextSplitter` from LangChain with separators optimized for Chinese and English: `["\n\n", "\n", "。", "！", "？", " ", ""]`.

**Step 3: Embedding Generation**

Each chunk is converted to a dense vector using `EmbeddingService` (`app/services/embedding_service.py`):

- **Model**: OpenAI `text-embedding-3-small` (default) or HuggingFace alternatives
- **Process**: `embed_batch(texts)` creates embeddings for all chunks in batches (default `batch_size = 100`)

For a chunk text `t`, the embedding vector is:

```
e(t) = EmbeddingModel(t) ∈ ℝ^d
```

Where `d` is the embedding dimension (1536 for `text-embedding-3-small`).

**Step 4: Vector Storage in Pinecone**

Embeddings are stored in a Pinecone index (`medical-documents`) via `PineconeService.upsert_vectors()` (`app/services/pinecone_service.py`):

- Each vector is associated with:
  - **Vector ID**: Unique identifier
  - **Metadata**: Chunk text, file path, chunk index, etc.
  - **Vector**: The embedding `e(t)`

**Replication**:
- Upload a medical document: `POST /documents/upload` with file
- Or process existing file: `POST /documents/process/blood_cancer_new.docx`
- Configure: `embedding_model="openai"`, `chunk_size=512`, `chunk_overlap=50`

### 3.5.2 Hematology Dictionary Vector Store

**Purpose**: Construct a specialized RAG index for hematology-specific terminology.

**Data Source**: `app/sources/data/hematology_dictionary.csv`

**Components**:
- `app/services/hematology_dictionary_loader.py` – loads CSV entries
- `app/services/medical_term_vector_store.py` – embeds and stores terms
- `app/services/hematology_retriever.py` – retrieves relevant entries

**Process**:

1. **Dictionary Loading**: CSV entries are loaded into memory
2. **Embedding**: Each dictionary entry (term + definition/example) is embedded using `EmbeddingService`
3. **Storage**: Vectors are stored in a dedicated Pinecone index (`hematology`)
4. **Retrieval**: `HematologyRetriever.retrieve_entries_for_correction()` uses medical terms from `query-index` to retrieve the most relevant dictionary entries

**Replication**: The hematology dictionary is typically pre-processed and stored. The retrieval process is automatic during correction.

---

## 3.6 Text Correction Methods

All correction methods share the same input (`Transcription.text`) and are evaluated using the same WER metric, enabling direct comparison. The methods are implemented in `app/services/llm.py` and exposed via `app/routes/audio.py`.

### 3.6.1 Baseline: Raw Whisper Transcription

- **Output**: `Transcription.text` (no additional processing)
- **Purpose**: Baseline for comparison
- **Storage**: Already stored during transcription (Section 3.4.1)

### 3.6.2 Method 1: Direct LLM Correction (No RAG)

**Endpoint**: `POST /audio/llm` with `use_rag = False` (default)

**Implementation**: `correct_whisper_text()` and `batch_correct_whisper_text()` in `app/services/llm.py`

**Process**:

1. **Prompt Construction**:
   ```
   base_prompt = "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
                 "1. 不補上標點符號\n"
                 "2. 只修正詞彙錯誤\n\n"
                 f"原文：{whisper_text}\n"
                 f"修正："
   ```

2. **LLM Generation**:
   ```python
   corrected_text = model_manager.generate_text(
       model_name=model_name,  # e.g., "gpt-4", "gpt-4o", "qwen2.5-7b-instruct"
       prompt=base_prompt,
       max_length=512,
       temperature=0.1
   )
   ```

3. **Storage**: Results stored in `LLMOutput.text` (with `text_with_rag = None`)

**Replication**:
```bash
POST /audio/llm
{
  "llm_model_name": "gpt-4",
  "prompt_version": "v1",
  "limit": 10,
  "use_rag": false
}
```

### 3.6.3 Method 2: Medical Document RAG + LLM

**Endpoint**: `POST /audio/llm` with `use_rag = True`

**Implementation**: Same functions as Method 1, but with RAG retrieval enabled

**RAG Retrieval Process** (`MedicalDocumentRetriever.retrieve_documents_for_correction` in `app/services/medical_document_retriever.py:30-110`):

**Step 1: Query Medical Terms**
- Retrieves stored medical terms for `transcription_id` from `query-index`:
  ```python
  transcription_embedding = embedding_service.embed_text(transcription_text)
  query_results = query_index.query(
      query_vector=transcription_embedding,
      top_k=top_k_queries * 2,
      filter={'transcription_id': {'$eq': transcription_id}, 'type': {'$eq': 'medical_term'}}
  )
  ```
- Extracts top `top_k_queries` unique terms

**Step 2: Document Retrieval**
- For each term, queries the `medical-documents` Pinecone index:
  ```python
  term_embedding = embedding_service.embed_text(term)
  doc_results = document_index.query(
      query_vector=term_embedding,
      top_k=top_k_documents,
      include_metadata=True
  )
  ```
- Retrieves top `top_k_documents` most similar document chunks per term

**Step 3: Context Assembly**
- Combines retrieved documents into context:
  ```
  context = "參考文檔 1：{doc_1}\n\n參考文檔 2：{doc_2}\n\n..."
  ```

**Step 4: Prompt Construction with RAG**
```
prompt = base_prompt + 
         f"以下是一些醫療文檔作為參考：\n{context}\n\n" +
         f"原文：{whisper_text}\n" +
         f"修正："
```

**Step 5: LLM Generation and Storage**
- Same as Method 1, but results stored in `LLMOutput.text_with_rag`

**Replication**:
```bash
POST /audio/llm
{
  "llm_model_name": "gpt-4",
  "prompt_version": "v1",
  "limit": 10,
  "use_rag": true,
  "top_k_queries": 3,
  "top_k_documents": 5
}
```

### 3.6.4 Method 3: Hematology Dictionary RAG + LLM

**Endpoint**: `POST /audio/llm/hematology`

**Implementation**: `correct_whisper_text_with_hematology()` and `batch_correct_whisper_text_with_hematology()` in `app/services/llm.py`

**Process**:

1. **Hematology Entry Retrieval** (`HematologyRetriever.retrieve_entries_for_correction` in `app/services/hematology_retriever.py:30-119`):
   - Uses medical terms from `query-index` to query the `hematology` Pinecone index
   - Retrieves top `top_k` dictionary entries per term

2. **Context Assembly**:
   ```
   context = "參考範例 1：{entry_1}\n\n參考範例 2：{entry_2}\n\n..."
   ```

3. **Prompt Construction**:
   ```
   prompt = base_prompt + 
            f"以下是一些血液學醫學詞典範例作為參考：\n{context}\n\n" +
            f"原文：{whisper_text}\n" +
            f"修正："
   ```

4. **Storage**: Results stored in `LLMOutput.text_with_hematology`

**Replication**:
```bash
POST /audio/llm/hematology
{
  "llm_model_name": "gpt-4",
  "prompt_version": "v1",
  "limit": 10,
  "top_k_queries": 2,
  "top_k": 5
}
```

---

## 3.7 Agent-Based Hybrid Correction

**Endpoint**: `POST /audio/llm/agent`

**Purpose**: An intelligent agent that dynamically selects and combines correction tools to achieve optimal quality.

**Implementation**:
- **Agent Orchestrator**: `TranscriptionAgent` (`app/services/transcription_agent.py`)
- **Available Tools**: Defined in `app/services/agent_tools.py`
- **Quality Evaluator**: `QualityEvaluator` (`app/services/quality_evaluator.py`)

### 3.7.1 Agent Tools

The agent has access to four correction tools (all implement the `AgentTool` interface):

1. **DirectLLMTool**: Direct LLM correction (equivalent to Method 1)
2. **MedicalDocumentRAGTool**: Medical document RAG (equivalent to Method 2)
3. **HematologyRAGTool**: Hematology dictionary RAG (equivalent to Method 3)
4. **CombinedRAGTool**: Simultaneously uses both medical documents and hematology dictionary as context

### 3.7.2 Agent Decision Loop

The agent follows an iterative refinement process (`TranscriptionAgent.correct_transcription`):

**Step 1: Initial Strategy Selection**

The agent selects an initial strategy based on transcription content:

```python
def _select_initial_strategy(whisper_text: str) -> str:
    text_lower = whisper_text.lower()
    hematology_keywords = ["血", "細胞", "骨髓", "淋巴", "白血球", "血小板"]
    if any(keyword in text_lower for keyword in hematology_keywords):
        return "hematology_rag"
    return "medical_document_rag"  # default
```

**Step 2: Iterative Correction with Quality Assessment**

For up to `max_iterations` (default: 3):

1. **Execute Current Tool**: `_execute_tool(tool_name, whisper_text, transcription_id, model_name, ...)`
2. **Evaluate Quality**: `QualityEvaluator.evaluate_correction_quality(original_text, corrected_text, method, metadata)`

**Quality Score Calculation** (`quality_evaluator.py:20-103`):

The quality score `Q` is computed as:

```
Q = 1.0 - Σ(penalties)
```

Where penalties are applied based on:

- **Length Deviation**: 
  - If `len(corrected) < 0.5 × len(original)`: penalty = 0.3 (truncation risk)
  - If `len(corrected) > 1.5 × len(original)`: penalty = 0.2 (hallucination risk)

- **No Change Detection**:
  - If `corrected.strip() == original.strip()`: penalty = 0.2

- **RAG Retrieval Quality** (if RAG was used):
  - If `documents_retrieved == 0`: penalty = 0.3
  - If `documents_retrieved < 2`: penalty = 0.1
  - Same logic applies to `entries_retrieved` for hematology RAG

The final score is normalized to [0, 1]:

```
Q_final = max(0.0, min(1.0, Q))
```

**Confidence and Recommendation**:

```
if Q_final >= 0.8:
    confidence = "high"
    recommendation = "accept"
elif Q_final >= 0.6:
    confidence = "medium"
    recommendation = "accept_or_refine"
else:
    confidence = "low"
    recommendation = "try_alternative"
```

3. **Decision Making**:
   - If `recommendation == "accept"`: Return current result as final
   - If `recommendation == "try_alternative"` and `len(methods_tried) < max_iterations`:
     - Get next method via `QualityEvaluator.suggest_next_method()`
     - Continue iteration

**Step 3: Method Suggestion Logic**

`QualityEvaluator.suggest_next_method()` implements a progressive strategy:

- If `direct_llm` tried → suggest `medical_document_rag`
- If `medical_document_rag` tried and text contains hematology keywords → suggest `hematology_rag`
- If 2+ methods tried → suggest `combined_rag`
- Otherwise: try next available method

**Step 4: Fallback and Result Selection**

- If all methods fail: Fallback to `direct_llm`
- If fallback fails: Return original text with error flag
- Otherwise: Return best-scoring result found

**Storage**: Final result stored in `LLMOutput.text_agent`

**Replication**:
```bash
POST /audio/llm/agent
{
  "llm_model_name": "gpt-4",
  "prompt_version": "v1",
  "limit": 10,
  "max_iterations": 3,
  "initial_strategy": null  # auto-select, or specify: "direct_llm", "medical_document_rag", "hematology_rag", "combined_rag"
}
```

---

## 3.8 Evaluation Metrics

### 3.8.1 Word Error Rate (WER) Calculation

**Purpose**: Quantitatively compare transcription and correction quality across all methods.

**Implementation**: `app/services/wer.py`

**Tokenization** (`clean_text()` function):

The system uses a **hybrid tokenization** approach to handle mixed Chinese-English text:

1. **Lowercase conversion**: `text = text.lower()`
2. **Punctuation removal**: All Chinese and English punctuation replaced with spaces
3. **Token extraction**:
   - **English tokens**: Word-level tokens matching pattern `[a-z0-9%]+` (alphanumeric sequences)
   - **Chinese tokens**: Character-level tokens (each Chinese character is a separate token)

The tokenization formula:

```
T = []
buffer = ""
for char in text:
    if char matches [a-z0-9%]:
        buffer += char
    else:
        if buffer:
            T.append(buffer)  # English word
            buffer = ""
        if char.strip():
            T.append(char)  # Chinese character
if buffer:
    T.append(buffer)
```

**WER Formula**:

WER is computed using the standard formula with the custom tokenizer:

```
WER = (S + D + I) / N
```

Where:
- `S` = number of substitutions
- `D` = number of deletions  
- `I` = number of insertions
- `N` = total number of tokens in reference

The implementation uses the `jiwer` library with custom tokenization:

```python
ref_tokens = clean_text(reference)
hyp_tokens = clean_text(hypothesis)
ref_str = " ".join(ref_tokens)
hyp_str = " ".join(hyp_tokens)
WER = jiwer.wer(ref_str, hyp_str)
```

WER values range from 0.0 (perfect match) to 1.0+ (many errors), with lower values indicating better performance.

### 3.8.2 WER Computation for All Methods

**Whisper WER**: `POST /audio/whisper_wer` (`app/routes/audio.py:81-114`)

- Compares `Transcription.text` (hypothesis) vs `Evaluation.ground_truth` (reference)
- Stores result in `Evaluation.whisper_wer`

**LLM WER**: `POST /audio/llm_wer` (`app/routes/audio.py:241-260+`)

For each `LLMOutput` with corresponding `Evaluation.ground_truth`:

- `LLMOutput.text` → `Evaluation.llm_wer`
- `LLMOutput.text_with_rag` → `Evaluation.llm_rag_wer`
- `LLMOutput.text_with_hematology` → `Evaluation.llm_hematology_wer`
- `LLMOutput.text_agent` → `Evaluation.llm_agent_wer`

All WER calculations use the same `wer()` function, ensuring **fair comparison** across methods.

**Replication**:
1. Ensure `Evaluation` records exist with `ground_truth` populated
2. Call `POST /audio/whisper_wer` to compute Whisper baseline WER
3. Call `POST /audio/llm_wer` to compute WER for all LLM correction methods

---

## 3.9 Implementation Details

### 3.9.1 Model Management

The system uses a unified model registry (`app/config/models.py`) that supports:

- **API Models**: OpenAI GPT-4, GPT-4o (accessed via API)
- **Local Models**: HuggingFace models (Qwen2.5, LLaMA 3, etc.) loaded locally
- **Model Manager**: `app/services/model_manager.py` handles loading, caching, and API calls

All LLM calls go through `model_manager.generate_text()`, ensuring consistent interface regardless of provider.

### 3.9.2 Vector Database

- **Platform**: Pinecone (managed vector database)
- **Indices**:
  - `query-index`: Stores extracted medical terms
  - `medical-documents`: Stores medical document chunks
  - `hematology`: Stores hematology dictionary entries
- **Embedding Model**: OpenAI `text-embedding-3-small` (1536 dimensions) by default
- **Similarity Metric**: Cosine similarity (Pinecone default)

### 3.9.3 Software Stack

- **Web Framework**: FastAPI (`app/main.py`)
- **ORM**: SQLAlchemy (`app/models/`, `app/repositories/`)
- **LLM APIs**: OpenAI Python SDK
- **Vector Database**: Pinecone Python SDK
- **Text Processing**: LangChain (`RecursiveCharacterTextSplitter`)
- **Evaluation**: `jiwer` library with custom tokenization
- **Audio Processing**: FFmpeg (via system calls)

---

## 3.10 Experimental Design

### 3.10.1 Controlled Variables

To ensure fair comparison, all methods share:

- **Input**: Same `Transcription.text` (Whisper baseline)
- **LLM Model**: Same model (configurable via `llm_model_name`, default: `"gpt-4"`)
- **Prompt Version**: Same prompt structure (configurable via `prompt_version`, default: `"v1"`)
- **Evaluation Metric**: Same WER calculation function
- **Ground Truth**: Same `Evaluation.ground_truth` for all methods

### 3.10.2 Independent Variables

- **Correction Method**: 
  - Baseline (Whisper only)
  - Direct LLM
  - Medical Document RAG
  - Hematology Dictionary RAG
  - Agent-based hybrid

- **RAG Parameters** (for RAG methods):
  - `top_k_queries`: Number of medical terms to use (default: 2-3)
  - `top_k_documents` / `top_k`: Number of retrieved documents/entries per term (default: 3-5)

- **Agent Parameters**:
  - `max_iterations`: Maximum correction attempts (default: 3)
  - `initial_strategy`: Starting method (auto-selected or manual)

### 3.10.3 Dependent Variables

- **WER**: Primary metric for all methods
- **Quality Score**: Agent's internal quality assessment (0.0-1.0)
- **Methods Tried**: Agent's exploration path (for analysis)

### 3.10.4 Replication Protocol

To replicate the entire experiment:

1. **Data Preparation**:
   ```bash
   POST /audio/parse {"input_dir": "origin", "output_dir": "parse"}
   POST /audio/split {"input_dir": "parse", "output_dir": "splited"}
   ```

2. **Transcription**:
   ```bash
   POST /audio/whisper/analyze {"input_dir": "splited", "extraction_model": "gpt-4o"}
   ```

3. **Knowledge Base Construction**:
   ```bash
   POST /documents/process/blood_cancer_new.docx
   # (Hematology dictionary pre-processed separately)
   ```

4. **Correction Methods** (in order):
   ```bash
   POST /audio/llm {"use_rag": false, "limit": 10}
   POST /audio/llm {"use_rag": true, "top_k_queries": 3, "top_k_documents": 5, "limit": 10}
   POST /audio/llm/hematology {"top_k_queries": 2, "top_k": 5, "limit": 10}
   POST /audio/llm/agent {"max_iterations": 3, "limit": 10}
   ```

5. **Evaluation**:
   ```bash
   POST /audio/whisper_wer
   POST /audio/llm_wer
   ```

6. **Analysis**: Query database to compare `Evaluation.whisper_wer`, `llm_wer`, `llm_rag_wer`, `llm_hematology_wer`, `llm_agent_wer`

---

## Summary

This methodology enables systematic comparison of multiple text correction approaches on the same dataset, with quantitative evaluation via WER. The agent-based method introduces adaptive tool selection, providing a hybrid approach that can outperform individual methods by dynamically choosing the best strategy for each transcription.
