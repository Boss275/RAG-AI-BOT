import streamlit as st
from rag_pipeline import load_documents, build_vectorstore, get_qa_chain
import os
import tempfile

st.set_page_config(page_title="📚 RAG AI Chatbot")
st.title("📚 RAG AI Chatbot")

# Load your OpenAI key from secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

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

    st.success("✅ Documents ready. Ask your questions below!")

    question = st.text_input("🤔 Your question")
    if question:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(question)
            st.write("🤖", answer)




pip install -q streamlit langchain openai faiss-cpu pypdf python-docx tiktoken
pip install -U langchain langchain-community google-generativeai faiss-cpu pypdf python-docx tiktoken
pip install -U langchain-community

import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile
from google.colab import files
from google.colab import userdata
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")
#os.environ["OPENAI_API_KEY"] = "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"

print("📁 Please upload your documents (PDF, TXT, or DOCX)...")
uploaded = files.upload()

file_paths = []
for filename in uploaded.keys():
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "wb") as f:
        f.write(uploaded[filename])
    file_paths.append(path)

def load_documents(file_paths):
    all_docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            print(f"❌ Skipped unsupported file type: {path}")
            continue
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

print("📄 Loading and parsing documents...")
documents = load_documents(file_paths)

print("🧠 Creating vector index...")
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(documents, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=vectordb.as_retriever()
)

print("✅ RAG Bot is ready! Ask questions based on your uploaded documents.")
print("❌ Type 'exit' to stop.")

while True:
    query = input("\n🤔 Your question: ")
    if query.lower().strip() == "exit":
        print("👋 Exiting. Thanks for using the RAG bot!")
        break
    answer = qa_chain.run(query)
    print("🤖", answer)
