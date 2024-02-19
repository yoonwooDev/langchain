# Q&A with RAG

**목차**
- [파이썬 소스코드](#파이썬-소스코드)
- [참고할 만한 사이트](#참고할-만한-사이트)
- [벡터데이터 생성하기](#벡터-데이터-생성하기)
- [벡터데이터 탐색하고 질의응답하기](#벡터-데이터-탐색-및-질의응답)

## 파이썬 소스코드
- `crete_pdf_vectorestore_faiss.py`: pdf 파일을 읽어와서 벡터 인덱스 데이터를 만들어 본인 컴퓨터에 index 데이터를 저장하는 역할을 하는 파이썬 코드
- `qa_pdf_vectorestore_faiss.py`: 위에서 생성한 index 데이터를 불러와서 질문에 답변하는
chatbot역할을 하는 파이썬 코드

## 참고할 만한 사이트
- https://python.langchain.com/docs/use_cases/question_answering/
- https://python.langchain.com/docs/modules/data_connection/vectorstores/
- https://iamajithkumar.medium.com/working-with-faiss-for-similarity-search-59b197690f6c
- https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf

## 벡터 데이터 생성하기
- 해당 소스 코드: `crete_pdf_vectorestore_faiss.py`
- 벡터 데이터베이스: facebook에서 만든 [FAISS](https://github.com/facebookresearch/faiss)를 사용한다.
- Text splitter: RecursiveCharacterTextSplitter를 사용하여 긴 텍스트를 작은 길이로 쪼갠다.

이 파일의 주 목적은 임의의 PDF 파일을 읽어와 openai에서 제공하는 `text-embedding-ada-002` 모델(https://platform.openai.com/docs/models/embeddings)을 이용하여 임베딩 벡터를 만들고 FAISS 데이터베이스를 로컬 노트북에 생성하여 임베딩 벡터 데이터를 저장한다. 

openai를 사용하기 위해 우선 `API_KEY`가 필요하다. 해당 키를 생성하기 위해서는 openai에 가입을 하고 카드 결제를 해야 한다 
- 가입하고 키 발급하는 방법: https://www.daleseo.com/chatgpt-api-keys/ 

그리고 발급받은 키를 코드에서 여러 방법으로 사용할 수 있다. 
- 발급 받은 키 사용방법: https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key 

이번 소스코드에서는 `.env`파일을 프로젝트 루트 디렉토리에 만들어서 아래와 같이 API_KEY를 설정하여 코드에서 사용한다. API_KEY값은 외부로 유출되지 않도록 조심해야한다. 유츌되어 다른 사람이 사용하면 요금을 내가 지불해야한다.

```python
API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

코드에서 사용하는 방법은 다음과 같다.
```python
import os
embeddings = OpenAIEmbeddings(api_key=os.environ.get("API_KEY", "<your OpenAI API key if not set as env var>"))  
```

아래 코드는 각종 변수를 선언하는 부분이다.

```python
faiss_db_path = "openai/examples/data/faiss_db/"
faiss_index_name = "chatpdf"
data_path = "openai/examples/data/"
training_data = data_path + "chosun-history.pdf"
```

여기서는 `chosun-history.pdf` 파일을 읽어와 벡터 데이터를 만들어 FAISS를 이용하여 로컬 필시에 저장할 것이다.

이제 `PyPDFLoader`를 이용해서 PDf파일을 로드한다.
```python
documents = PyPDFLoader(training_data).load()
```
VSCODE 디버거를 이용해서 반환되는 `documents`를 살펴 보면 아래 그림과 같이 `Document` 타입의 객체로 구성된 배열이라는 것을 알 수 있다. PyPDFLoader.load()함수는 각 페이지마다 Document객체를 생성하여 리턴한다.

![img](images/document.png)

`chosun-history.pdf` 파일은 총 7장으로 구성되어 있고 각 페이지의 내용이 `Document`객체로 만들어져 있다. `Document`객체는 `page_content`와 metadata로 구성된 객체이다. 그런데 한 페이지마다 글자수가 너무 많아 openai model에 모두 입력되지 않는 경우가 있다. 각각의 openai model은 한번에 처리할 수 있는 토큰수가 정해져 있다. (참고: https://platform.openai.com/docs/models/models) 따라서, 한 페이지 분량의 글자를 작은 단위로 쪼개는 작업이 필요하고 쪼개는 작은 단위를 각각 chunk라고 한다. 아래 코드는 랭체인에서 제공하는 splitter를 이용해서 documents를 chunk단위로 쪼개는 작업을 수행한다. 1개의 chunk는 최대 1000개의 문자를 포함하며 각각의 chunk마다 200개 문자를 중첩해서 가지도록 설정하였다. (LLM이 좀 더 연관있는 chunk를 구분하고 탐색하기 위해)

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                               chunk_overlap=200,
                                               add_start_index=True)
texts = text_splitter.split_documents(documents)
```

디버거로 리턴값 `texts`를 확인해 보면 아래와 같다.

![img](images/split_texts.png)

위에서 documents와 비교해 보면 더 작은 단위로 쪼개져 document 객체가 12개로 늘어난 것을 볼 수 있다. 이제 잘게 쪼갠 chunk데이터를 embedding 모델에 입력으로 넣어 주어 임베딩 벡터 데이터를 만들도록 한다. 그리고 FAISS 데이터베이스를 생성하여 만들어진 벡터 데이터를 로컬 피시에 저장한다.

```python
embeddings = OpenAIEmbeddings(api_key=os.environ.get("API_KEY", "<your OpenAI API key if not set as env var>"))  
 
try:
    faiss_db = FAISS.from_documents(texts, embeddings)    
    faiss_db.save_local(faiss_db_path, index_name=faiss_index_name)
    print("Faiss db created")
except Exception as e:
    print("Faiss store failed \n", e)
```
소스코드에서 지정한 경로에 실제 faiss db가 생성되었는지 확인해 본다.

## 벡터 데이터 탐색 및 질의응답