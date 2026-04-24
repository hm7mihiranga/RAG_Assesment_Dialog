# Dialog Assignment Project - MH Mihiranga

A minimal document Q&A application with:

- **FastAPI backend** for ingesting documents and answering questions
- **Streamlit UI** for uploading files, pasting text, and chatting
- **FAISS** for vector search
- **EMBEDDING** used the Hugging-Face Model sentence-transformers/all-MiniLM-L6-v2
- **RE-RANKING** used the Hugging-Face Model cross-encoder/ms-marco-MiniLM-L-6-v2
- **Gemini** for generating answers (Model: gemini-3-flash-preview)

## Run locally with Docker

### 1. Prerequisites

- Docker Desktop installed and running
- A valid Google Gemini API key

### 2. Create your `.env`

Create a file named `.env` in the project root and add your key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Start the app

From the project root, run:

```bash
docker compose up --build
```

This will start:

- FastAPI backend at `http://localhost:8010`
- Streamlit UI at `http://localhost:8501`

### 4. Use the app

1. Open `http://localhost:8501` 
2. Upload a `.txt` or `.pdf`, or paste text into the app
3. Ask questions in the chat panel
4. Added Conversational Memory also, so limit it for 3 conversations, from using conversations you can add the context knowledge
4. Use **Close Session** to clear backend chat memory

### 5. Stop the containers

```bash
docker compose down
```

## API documentation (FASTAPI - Checking)

When the backend is running, open:

- `http://localhost:8010/docs`

## Notes

- The UI keeps a `conversation_id` in browser session state for follow-up questions.
- The backend uses `artifacts/` for the FAISS index and metadata (Use parquet files instead of CSV, becuase quick adaptation).

## CI/CD and Deployment

- **CI/CD**: GitHub Actions builds the Docker image on every push to `main`.
- **Deployment**: The built image is pushed to Google Artifact Registry and deployed to Google Cloud Run.
- **Cost**: Use Cloud Run with `min-instances=0` and `max-instances=1` for low-cost scale-to-zero hosting.
- **Services**: Deploy the FastAPI backend and Streamlit UI as separate Cloud Run services.
