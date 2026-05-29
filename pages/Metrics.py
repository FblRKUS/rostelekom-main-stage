import json
from pathlib import Path
import streamlit as st
import pandas as pd  # type: ignore[import]
from vector_store import VectorStore
from score import score_question

st.set_page_config(page_title="RAG Metrics", layout="wide")
st.title("Evaluation Metrics")
st.markdown("Calculate Precision@5 against the evaluation dataset.")


@st.cache_resource
def get_store():
    return VectorStore()


store = get_store()

with st.sidebar:
    st.header("Evaluation Settings")
    eval_file = st.text_input(
        "eval_questions.json path",
        value="docs/internal/dataset_case3_v1.0_fix/eval_questions.json",
    )
    alpha = st.slider(
        "Hybrid Search Weight (Alpha)",
        0.00,
        1.00,
        0.75,
        0.01,
        help="1.0 = Vector, 0.0 = BM25",
    )

if st.button("Run Evaluation", type="primary"):
    q_path = Path(eval_file)
    if not q_path.exists():
        st.error(f"File not found: {eval_file}")
    else:
        with open(q_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        with st.spinner(f"Evaluating {len(questions)} questions..."):
            per_question = []
            for q in questions:
                qid = q["question_id"]
                query = q["query"]
                correct = q.get("correct_chunk_ids", [])

                # Use hybrid search
                results = store.hybrid_search(query, top_k=5, alpha=alpha)
                top5 = [r.chunk_id for r in results]

                score = score_question(top5, correct)
                per_question.append(
                    {
                        "Question ID": qid,
                        "Query": query,
                        "Difficulty": q.get("difficulty", "unknown"),
                        "Language": q.get("language", "unknown"),
                        "Precision@5": score,
                    }
                )

            df = pd.DataFrame(per_question)
            mean_score = df["Precision@5"].mean()

            st.success("Evaluation complete!")
            st.metric(label="Mean Precision@5", value=f"{mean_score:.2%}")

            st.markdown("### Per-question Details")
            st.dataframe(
                df.style.format({"Precision@5": "{:.2%}"}),
                width="stretch",
                hide_index=True,
            )
