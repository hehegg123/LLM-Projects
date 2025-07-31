from langchain.chains.combine_documents.stuff import StuffDocumentsChain, create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
import os

from pydantic import model_validator
from pydantic.v1 import Field
from pydantic.v1 import BaseModel
from typing import Literal

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

#1. stuff
# load initial web article

loader = WebBaseLoader('https://lilianweng.github.io/posts/2025-05-01-thinking/')
docs = loader.load()

prompt = ChatPromptTemplate.from_template(
    """
    write a summary based on the document provided below
    {context}
    摘要:
    """
)
stuff_chain = create_stuff_documents_chain(model, prompt)
result = stuff_chain.invoke({'context':docs})
print(result)
#
# chain = prompt | model
# result = chain.invoke({'text':docs})
# print(result)



