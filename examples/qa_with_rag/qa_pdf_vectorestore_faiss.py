# https://python.langchain.com/docs/use_cases/question_answering/
# https://python.langchain.com/docs/use_cases/question_answering/quickstart
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# models (refer to https://platform.openai.com/docs/models/embeddings)
# GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL = "gpt-4"

faiss_db_path = "examples/data/faiss_db/"
faiss_index_name = "chatpdf"

embeddings = OpenAIEmbeddings(api_key=os.environ.get("API_KEY", "<your OpenAI API key if not set as env var>"))    

try:
    faiss_db = FAISS.load_local(faiss_db_path, embeddings, index_name=faiss_index_name)
    print("Faiss db loaded")
except Exception as e:
    print("Faiss db loading failed \n", e)

retriever = faiss_db.as_retriever()

template = """아래의 context 내용을 기반으로 답변합니다. 만약 context에서 내용을
찾을 수 없다면 "죄송합니다. 답변을 찾을 수가 없습니다."로 답변을 합니다.
:{context}

Question: {question}
""" 

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(api_key=os.environ.get("API_KEY", "<your OpenAI API key if not set as env var>"),
                 model=GPT_MODEL)

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    user_question = input("Question (Type 'q' to quit): ")
    if user_question.lower() == 'q':
        break
    # print(chain.invoke("What is the number of training tokens for LLaMA2?")) 
    print(chain.invoke(user_question)) 
