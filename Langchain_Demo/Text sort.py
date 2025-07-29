from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from pydantic.v1 import Field
from pydantic.v1 import BaseModel

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20', temperature = 0)

class Classification(BaseModel):
    sentiment: str = Field (description = 'sentiment of text')
    aggressiveness: int = Field(description="describe the aggressiveness of the text from 1 - 10")
    language: str = Field(description="the language used by the text")