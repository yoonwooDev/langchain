# https://python.langchain.com/docs/modules/data_connection/vectorstores/
# https://iamajithkumar.medium.com/working-with-faiss-for-similarity-search-59b197690f6c
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

faiss_db_path = "openai/examples/data/faiss_db/"
faiss_index_name = "chatpdf"
data_path = "openai/examples/data/"
training_data = data_path + "chosun-history.pdf"

documents = PyPDFLoader(training_data).load()
print(f"{len(documents)} documents")
print(f"{len(documents[0].page_content)} characters in the documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                               chunk_overlap=200,
                                               add_start_index=True)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(api_key=os.environ.get("API_KEY", "<your OpenAI API key if not set as env var>"))  
 
try:
    faiss_db = FAISS.from_documents(texts, embeddings)    
    faiss_db.save_local(faiss_db_path, index_name=faiss_index_name)
    print("Faiss db created")
except Exception as e:
    print("Faiss store failed \n", e)