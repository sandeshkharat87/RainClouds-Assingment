#importing libaries
from dotenv import load_dotenv
load_dotenv("/home/wpnx/.secrets/.env")

from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings,SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
import pandas as pd
import requests
from langchain.llms import Ollama
from langchain.agents import tool, initialize_agent,AgentExecutor, create_react_agent
from langchain import hub


### setting up llm and embedding model
# llm = ChatGroq(name="llama3.1")
llm = Ollama(model= "gemma2:2b")
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


main_database = FAISS.load_local("rainclouds-v1", embed_model,allow_dangerous_deserialization=True)


## Prompt for CIS and AWS GCP chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

cis_template = """
You are a bot for Center for Internet Security (CIS). You have to 
answer CIS Questions. If you dont know answer just say i dont know
Question:{question}
Context:{context}
Answer
"""


env_temp="""
You are helful bot for Center for Internet Security (CIS), Amazons AWS and Googles GCP cloud Information. which answers users query from the give context.
From given context find the answers if answer not in the context just say I dont know , nothing else.
Question:{question}
Context:{context}
Answer:
"""

env_prompt = PromptTemplate.from_template(env_temp)
# agent_prompt = hub.pull("hwchase17/react")
# cis_prompt = PromptTemplate.from_template(cis_template)

from langchain.chains import RetrievalQA

env_chain = RetrievalQA.from_chain_type(llm=llm,retriever=main_database.as_retriever() ,chain_type_kwargs={"prompt":env_prompt})
# cis_chain = RetrievalQA.from_chain_type(llm=llm,retriever=cis_db.as_retriever() ,chain_type_kwargs={"prompt":cis_prompt},return_source_documents=True)


def chat(question,hist):
    answer = env_chain.invoke(question)["result"]
    # return {"question":question, "answer":answer}
    return answer