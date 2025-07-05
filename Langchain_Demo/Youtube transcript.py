from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import datetime

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

persist_dir = 'chroma_data_dir' # directory for Vector Database

# urls = ["https://www.youtube.com/watch?v=LCEmiRjPEtQ",
#         "https://www.youtube.com/watch?v=PL5QnLrOjqk",
#         "https://www.youtube.com/watch?v=FwOTs4UxQS4"]
#
# docs = []
# for url in urls:
#     try:
#         docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info= False).load())
#         print(f"Successfully loaded content from: {url}")
#     except Exception as e:
#         print(f"Error loading content from {url}: {e}")
# print(len(docs))
#
# #for doc in docs:
#     #doc.metadata['publish_year'] = int(datetime.datetime.strptime(doc.metadata['publish_date'],'%Y-%m-%d %H:%M:%S').strftime('%Y'))
#
# #print(docs[[0].metadata])
# print(docs[0])
#
# # create the vector database based on multiple document
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=30)
# split_doc = text_splitter.split_documents(docs)
#
# vectorstore = Chroma.from_documents(split_doc ,GoogleGenerativeAIEmbeddings,persist_directory = persist_dir)

vectorstore = Chroma(persist_directory =persist_dir, embedding_function = GoogleGenerativeAIEmbeddings)

result = vectorstore.similarity_search('what is the current status of current agent Role?')
print(result[0])