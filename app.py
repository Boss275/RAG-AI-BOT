import streamlit as st
from rag_pipeline import load_documents, build_vectorstore, get_qa_chain
import os
import tempfile

st.title("📚 RAG AI Chatbot")

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

uploaded_files = st.file_uploader("Upload your documents (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as temp_file:
            temp_file.write(file.read())
            file_paths.append(temp_file.name)

    with st.spinner("Loading and indexing documents..."):
        docs = load_documents(file_paths)
        vectordb = build_vectorstore(docs)
        qa_chain = get_qa_chain(vectordb)

    st.success("Documents ready. Ask your questions below!")

    question = st.text_input("🤔 Your question")
    if question:
        answer = qa_chain.run(question)
        st.write("🤖", answer)
