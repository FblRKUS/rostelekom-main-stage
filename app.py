import streamlit as st
from rag_pipeline import RAGAnswerGenerator
from index import index_repository
from shared import get_store

st.set_page_config(page_title="CodeLens RAG", layout="wide")

st.title("CodeLens RAG")
st.markdown("Search code using natural language.")


@st.cache_resource
def get_rag(use_llm):
    return RAGAnswerGenerator(use_llm=use_llm)


store = get_store()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("Generate answer using LLM (Ollama)", value=False)
    alpha = st.slider(
        "Hybrid Search Weight",
        min_value=0.00,
        max_value=1.00,
        value=0.75,
        step=0.01,
        help="1.0 = Vector search only, 0.0 = Keyword search only (BM25).",
    )

    st.divider()
    st.subheader("Index Codebase")
    index_input = st.text_input(
        "Local path or GitHub URL",
        placeholder="./my_project  or  https://github.com/owner/repo",
    )
    if st.button("Index", use_container_width=True):
        if not index_input.strip():
            st.warning("Enter a path or GitHub URL.")
        else:
            with st.spinner("Indexing..."):
                src = index_input.strip()
                is_github = src.startswith("http")
                try:
                    result = index_repository(
                        path=None if is_github else src,
                        github=src if is_github else None,
                    )
                    get_store.clear()
                    if result.startswith("Error"):
                        st.error(result)
                    else:
                        st.success(result)
                except Exception as e:
                    st.error(f"Error: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("results"):
            st.markdown("### Search Results")
            for i, res in enumerate(message["results"], 1):
                with st.expander(
                    f"[{i}] {res['path']} | {res['type']}: {res['name']} (Relevance: {res['relevance']:.1f}%)"
                ):
                    st.code(res["content"], language="python")

query = st.chat_input("Enter your query (e.g. как обработать ошибку?)")

if query:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        st.markdown("### Search Results")
        results = store.hybrid_search(query, top_k=5, alpha=alpha)

        results_data = []
        if not results:
            st.info("No relevant code fragments found.")
            answer = "No relevant code fragments found."
        else:
            for i, res in enumerate(results, 1):
                path = res.metadata.get("file_path", "Unknown")
                type_ = res.metadata.get("type", "Unknown")
                name = res.metadata.get("name", "Unknown")
                relevance = res.score * 100.0

                results_data.append(
                    {
                        "path": path,
                        "type": type_,
                        "name": name,
                        "relevance": relevance,
                        "content": res.content,
                    }
                )

                with st.expander(
                    f"[{i}] {path} | {type_}: {name} (Relevance: {relevance:.1f}%)"
                ):
                    st.code(res.content, language="python")

            if use_llm:
                st.markdown("### LLM Answer")
                rag = get_rag(use_llm=True)
                if not rag.use_llm:
                    st.warning("Ollama is not available or model is missing.")
                    answer = "Ollama is not available."
                else:
                    with st.spinner("Generating answer..."):
                        placeholder = st.empty()
                        answer = ""

                        history = [
                            m
                            for m in st.session_state.messages[:-1]
                            if m["role"] in ("user", "assistant")
                        ]

                        for chunk in rag.generate_stream(
                            query, results, history=history
                        ):
                            answer += chunk
                            placeholder.markdown(answer)
            else:
                answer = "Found relevant fragments."

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "results": results_data}
        )
