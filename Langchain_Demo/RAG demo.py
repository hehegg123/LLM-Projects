from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
import bs4
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import chat_agent_executor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

# 1. load data
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2024-11-28-reward-hacking/'],
    bs_kwargs= dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title','post-content'))
    )# select the label we want it to read from

)

docs = loader.load()

# 2. Text splitters to split the document to smaller pieces. (for large documents)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5)

splits = splitter.split_documents(docs)

# 3. store in vector
embeddings_google = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma.from_documents(documents = splits, embedding = embeddings_google)

# 4. create the retriever for search
retriever = vector_store.as_retriever()

# 5. create a template for prompt
system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the questions.
If you don't know the answer, say that you don't know, Use concise answers with in three sentences.\n
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 6. Create Chain
chain1 = create_stuff_documents_chain(model, prompt) # connect prompt to model to send question to LLM
# chain2 = create_retrieval_chain(retriever, chain1) # connect the retriever to chain 1 providing it's ability to search

# 7. create a subchain for sending history context for retriever.
contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in 
the chat history formulate a standalone question which can be understood without the chat history. Do NOT answer the question,
just reformuate it if needed nad otherwise return is as is.
"""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}")
    ]
)

history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

#save chat history
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# assembly the mean chain
chain = create_retrieval_chain(history_chain,chain1)

result_chain = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

# Test 1
response1 = result_chain.invoke(
    {'input': 'what is reward hacking'},
    {'configurable': {'session_id': 'hehegg123'}}
)
print(response1['answer'])

#test 2
response2 = result_chain.invoke(
    {'input':"what is it's affect?"},
    {'configurable': {'session_id': 'hehegg123'}}
)
print(response2['answer'])