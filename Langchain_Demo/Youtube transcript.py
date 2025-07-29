
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
persist_dir = 'chroma_data_dir' # directory for Vector Database

urls = ["https://www.youtube.com/watch?v=LCEmiRjPEtQ",
        "https://www.youtube.com/watch?v=PL5QnLrOjqk",
        "https://www.youtube.com/watch?v=FwOTs4UxQS4"]

docs = []
for url in urls:
    try:
        docs.extend(YoutubeLoaderDL.from_youtube_url(url, add_video_info= True).load())
        print(f"Successfully loaded content from: {url}")
    except Exception as e:
        print(f"Error loading content from {url}: {e}")

print(docs[0])
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
# vectorstore = Chroma.from_documents(split_doc ,embeddings_model,persist_directory = persist_dir)
#
#
# vectorstore = Chroma(persist_directory =persist_dir, embedding_function = embeddings_model)
# #
# # result = vectorstore.similarity_search('what is the current status of current agent Role?')
# # print(result[0])
#
# system = """
# You are an expert at converting user questions into database queries.
# You have access to a database of tutorial videos about a software library for building LLM-powered applications.
# Given a question, return a list of database queries optimized to retrieve the most relevant results.
#
# If there are acronyms or words you are not familiar with, do not try to rephrase them.
# """
#
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system",system),
#         ("human","{question}"),
#     ]
# )
#
# #pydantic
# class Search(BaseModel):
#     # search based on similarity
#     query: str = Field(None, description ="Similarity search query")
#
# chain = {'question': RunnablePassthrough()} | prompt | model.with_structured_output(Search)
#
# response = chain.invoke("How to build an agent?")
# print(response)
#
# def retrieval(search: Search)-> List[Document] :
#     _filter = None
#     return vectorstore.similarity_search(search.query, filter=_filter)
#
# new_chain = chain | retrieval
#
# result = new_chain.invoke("How to build an agent?")
# print(result)

