import os
import requests
import streamlit as st
from requests.exceptions import RequestException


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="R",
    layout="wide",
)

st.markdown(
    """
    <style>
      .hero {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
      }
      .muted {
        color: #64748b;
        font-size: 0.92rem;
      }
      .source-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        background: #ffffff;
        margin-bottom: 0.45rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "api_base": os.getenv("API_BASE", "http://127.0.0.1:8010"),
        "top_k": 0,
        "conversation_id": None,
        "messages": [],
        "health": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def api_get(path: str):
    try:
        response = requests.get(f"{st.session_state.api_base}{path}", timeout=20)
        body = response.json() if response.content else {}
        return response.status_code, body
    except RequestException as exc:
        return None, {"error": str(exc)}


def api_post_json(path: str, payload: dict):
    try:
        response = requests.post(f"{st.session_state.api_base}{path}", json=payload, timeout=120)
        body = response.json() if response.content else {}
        return response.status_code, body
    except RequestException as exc:
        return None, {"error": str(exc)}


def api_post_file(path: str, file_obj, source: str):
    files = {"file": (file_obj.name, file_obj.getvalue(), file_obj.type or "application/octet-stream")}
    data = {}
    if source.strip():
        data["source"] = source.strip()

    try:
        response = requests.post(
            f"{st.session_state.api_base}{path}",
            files=files,
            data=data,
            timeout=180,
        )
        body = response.json() if response.content else {}
        return response.status_code, body
    except RequestException as exc:
        return None, {"error": str(exc)}


def clear_local_state() -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources returned.")
        return

    for source in sources:
        src = source.get("source", "unknown")
        page = source.get("page", 0)
        chunk = source.get("chunk_id", "-")
        score = source.get("score", "-")
        rerank = source.get("rerank_score", "-")
        st.markdown(
            f"""
            <div class="source-card">
              <b>{src}</b><br/>
              <span class="muted">page={page} | chunk={chunk} | vector={score} | rerank={rerank}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


init_state()

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">RAG Document Assistant</h2>
      <div style="opacity:0.9; margin-top:4px;">Upload one document set, ask questions, and keep short chat memory per conversation.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Connection")
    st.session_state.api_base = st.text_input("API Base URL", value=st.session_state.api_base)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Health", use_container_width=True):
            st.session_state.health = api_get("/health")
    with col_b:
        if st.button("Close Session", use_container_width=True):
            code, body = api_post_json("/session/close", {})
            if code == 200:
                clear_local_state()
                st.success("Backend session cleared.")
            else:
                clear_local_state()
                st.warning(f"Close endpoint response: {code} {body}")

    st.subheader("Ask Settings")
    st.session_state.top_k = st.slider("top_k", 1, 12, int(st.session_state.top_k))

    st.caption("conversation_id")
    st.code(str(st.session_state.conversation_id), language=None)

    if st.session_state.health:
        code, body = st.session_state.health
        if code == 200:
            st.success(f"OK | chunks={body.get('chunks', 0)} | sessions={body.get('sessions', 0)}")
        else:
            st.error(f"Health failed: {code} {body}")

left, right = st.columns([1.0, 1.35])

with left:
    st.subheader("Ingest")
    tabs = st.tabs(["Upload File", "Paste Text"])

    with tabs[0]:
        uploaded = st.file_uploader("Choose .txt or .pdf", type=["txt", "pdf"])
        file_source = st.text_input("Optional source label", value="")
        if st.button("Ingest File", type="primary", use_container_width=True):
            if uploaded is None:
                st.error("Please select a file.")
            else:
                with st.spinner("Uploading and indexing..."):
                    code, body = api_post_file("/ingest", uploaded, file_source)
                if code == 200:
                    clear_local_state()
                    st.success(f"Ingested: {body}")
                else:
                    st.error(f"Ingest failed: {code} {body}")

    with tabs[1]:
        raw_text = st.text_area("Text", height=180, placeholder="Paste document text here...")
        text_source = st.text_input("Text source", value="manual-text")
        if st.button("Ingest Text", use_container_width=True):
            if not raw_text.strip():
                st.error("Text is empty.")
            else:
                payload = {"text": raw_text, "source": text_source or "manual-text"}
                with st.spinner("Indexing text..."):
                    code, body = api_post_json("/ingest", payload)
                if code == 200:
                    clear_local_state()
                    st.success(f"Ingested: {body}")
                else:
                    st.error(f"Ingest failed: {code} {body}")

with right:
    st.subheader("Chat")

    if not st.session_state.messages:
        st.info("Ask about the currently uploaded document.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources", expanded=False):
                    render_sources(msg["sources"])

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload = {"question": prompt, "top_k": int(st.session_state.top_k)}
        if st.session_state.conversation_id:
            payload["conversation_id"] = st.session_state.conversation_id

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                code, body = api_post_json("/ask", payload)

            if code == 200:
                answer = body.get("answer", "")
                sources = body.get("sources", [])
                cid = body.get("conversation_id")
                if cid:
                    st.session_state.conversation_id = cid

                st.markdown(answer or "No answer returned.")
                if sources:
                    with st.expander("Sources", expanded=False):
                        render_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer or "", "sources": sources}
                )
            else:
                err = f"Request failed: {code} {body}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
