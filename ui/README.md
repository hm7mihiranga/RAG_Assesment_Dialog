# Streamlit UI (Minimal)

## File structure

- `ui/app.py` - Streamlit UI for ingest + ask
- `ui/README.md` - run instructions

## Run

1. Start FastAPI backend:

   `uvicorn src.api:app --reload --port 8010`

2. Start Streamlit UI:

   `streamlit run ui/app.py`

3. Open browser:

   `http://localhost:8501`

## Notes

- UI keeps `conversation_id` in session state for follow-up questions.
- Uploading a new file or text resets local chat state.
- `Close Session` calls `/session/close` and clears backend runtime state.
