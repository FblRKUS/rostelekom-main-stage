import streamlit as st
from vector_store import VectorStore


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()
