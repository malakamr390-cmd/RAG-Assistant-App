import streamlit as st
from api_client import ask_question

st.set_page_config(page_title="RAG Lecture Assistant", page_icon="📚")

st.title("📚 RAG Lecture Assistant")
st.caption("Ask questions about BUA 304: Predictive Analytics lectures")

question = st.text_input("Ask a question about the lectures:")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            result = ask_question(question)
            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### Sources")
            for source in result["sources"]:
                st.write(f"- {source}")
        except Exception as e:
            st.error(f"Something went wrong while contacting the backend: {e}")