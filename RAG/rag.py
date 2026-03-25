import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# from langchain.chains.retrieval_qa.base import RetrievalQA
load_dotenv()
model=ChatOpenAI(model="openrouter/free",api_key=os.getenv("OPENAI_API_KEY"),base_url="https://openrouter.ai/api/v1")

# Loading Document
loader=UnstructuredMarkdownLoader("RAG/doc/info.md")
document=loader.load()
# print(document[0].page_content[:50])

# Chunking
t_split=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
splitted_doc=t_split.split_documents(document)
# print(splitted_doc)
# print(f"Total chunks: {len(splitted_doc)}")


# Embedding
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore=FAISS.from_documents(splitted_doc,embedding)


system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always be concise and provide accurate answers. If no context is recieved and query is general you can answer fron your own knowledge base else if context is recieved always answer based on context """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{system_prompt}\nContext:\n{{context}}\nQuestion:\n{{question}}"
)


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)
qa_chain=RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)
# query="what is my name?"
# result=qa_chain.invoke({"query":query})
# print(result)
# print(query)
# print("Result",result["result"])
while True:
    inp=input("Query : ")
    if inp=="0":
        break
    result=qa_chain.invoke({"query":inp})
    print("Result",result["result"])
