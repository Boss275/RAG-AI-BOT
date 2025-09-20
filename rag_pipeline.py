from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

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
            continue
        all_docs.extend(loader.load())
    return all_docs

def build_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb

def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
