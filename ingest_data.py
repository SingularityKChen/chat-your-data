import pickle

import faiss
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

import langchain
langchain.debug = True
# Load default environment variables (.env)
load_dotenv("env.txt")
# Here we load in the data in the format that Notion exports it in.
folder_name = "source"
loader = PyPDFDirectoryLoader(folder_name + "/")
pages = loader.load()
if len(pages) > 0:
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(pages)
    # Here we create a vector store from the documents and save it to disk.
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)
    faiss.write_index(store.index, "docs.index")
    store.index = None
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(store, f)
