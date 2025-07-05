from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

import os


os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
embeddings_google = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# prepare testing data:
documents = [
    Document(
        page_content="Dog is a great companion for it's loyal.",
        metadata={'source':'Memo Pet documents'},
    ),
    Document(
        page_content="Cat is a great companion because it's easy to take care of.",
        metadata = {'source':'Memo Pet documents'},
    ),
    Document(
        page_content="Cow is a great companion for it's produces.",
        metadata = {'source':'Memo Pet documents'},
    )
]

# tensor database
vector_store = Chroma.from_documents(documents, embedding = embeddings_google)

#Similarity Check: return a similar score, lower the similar score, higher the similarity
#print(vector_store.similarity_search_with_score('easy to take care of'))

#establish search engine, return with highest similarly with bind(k=1)
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

#print(retriever.batch(['coffee cat','milk provider']))

message = """
use provided context to answer the following question.
{question}
context:
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

chain = {'question': RunnablePassthrough(),'context': retriever} | prompt_temp | model

response = chain.invoke('please tell me what is a cat?')
print(response.content)

