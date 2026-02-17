# Agent Implementation for Medical Transcription Correction

## Overview

An intelligent agent system has been implemented to dynamically select and combine correction tools for medical transcriptions. The agent can choose between different strategies (direct LLM, medical document RAG, hematology RAG, or combined RAG) based on quality assessments.

## Architecture

### Components

1. **Agent Tools** (`app/services/agent_tools.py`)
   - `DirectLLMTool`: Fast correction without retrieval
   - `MedicalDocumentRAGTool`: Uses medical documents from Pinecone
   - `HematologyRAGTool`: Uses hematology dictionary
   - `CombinedRAGTool`: Combines both medical documents and hematology dictionary

2. **Quality Evaluator** (`app/services/quality_evaluator.py`)
   - Evaluates correction quality
   - Suggests next method if quality is low
   - Determines confidence levels

3. **Transcription Agent** (`app/services/transcription_agent.py`)
   - Orchestrates the correction process
   - Selects initial strategy
   - Iterates if quality is low
   - Returns best result

## Database Changes

### New Columns

- `llm_outputs.text_agent`: Stores agent-corrected text
- `evaluations.llm_agent_wer`: Stores WER for agent corrections

### Migration

Run the migration to add the new columns:
```bash
alembic upgrade head
```

## API Usage

### Endpoint: `/audio/llm/agent`

**POST** request to correct transcriptions using the agent.

**Parameters:**
- `llm_model_name` (str, default: "gpt-4"): LLM model to use
- `prompt_version` (str, default: "v1"): Prompt version
- `limit` (int, default: 10): Maximum transcriptions to process
- `max_iterations` (int, default: 3): Maximum correction attempts per transcription
- `initial_strategy` (str, optional): Initial strategy to try
  - Options: `"direct_llm"`, `"medical_document_rag"`, `"hematology_rag"`, `"combined_rag"`
  - If None, agent auto-selects based on transcription content

**Example:**
```bash
POST /audio/llm/agent?llm_model_name=gpt-4&limit=5&max_iterations=3
```

## How It Works

1. **Initial Strategy Selection**
   - Agent analyzes transcription text
   - Checks for hematology-specific keywords
   - Selects appropriate initial strategy (default: medical_document_rag)

2. **Tool Execution**
   - Agent executes selected tool
   - Retrieves relevant documents/entries if using RAG
   - Generates corrected text using LLM

3. **Quality Assessment**
   - Evaluates correction quality based on:
     - Text length changes
     - Whether changes were made
     - Number of documents/entries retrieved
   - Assigns confidence level (high/medium/low)

4. **Iteration Decision**
   - If quality is high → Accept and return
   - If quality is medium → Accept or try alternative
   - If quality is low → Try alternative method

5. **Result Storage**
   - Stores best result in `text_agent` column
   - Includes metadata about methods tried and quality scores

## Comparison with Previous Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Direct LLM** | Fast, simple | May miss medical terminology |
| **Medical RAG** | Good for general medical terms | May not cover specialized areas |
| **Hematology RAG** | Best for hematology terms | Limited to hematology |
| **Agent** | Adapts to content, tries multiple strategies | Slower, more API calls |

## Best Practices

1. **For Simple Cases**: Use direct LLM or fixed RAG approach (faster)
2. **For Complex Cases**: Use agent (better quality)
3. **For Batch Processing**: Start with agent on a small sample, then decide
4. **Cost Optimization**: Set `max_iterations=2` to limit API calls

## Evaluation

After running agent corrections, calculate WER:
```bash
POST /audio/llm_wer
```

This will calculate WER for all methods including agent (`llm_agent_wer`).

## Future Enhancements

- Add more specialized tools (e.g., oncology dictionary)
- Improve quality evaluation with ML models
- Add caching for repeated transcriptions
- Support parallel processing for batch operations
