
from typing import List, Optional

from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pydantic.v1 import Field
from pydantic.v1 import BaseModel

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#define person class
class Person(BaseModel):
    name: Optional[str] = Field(default = None, description="Person's name")

    hair_color: Optional[str] = Field(default = None, description="Person's hair color")

    height_in_meters: Optional[float] = Field(default = None, description="Person's height")

class ManyPerson(BaseModel):
    people: List[Person]


# define prompt template to define the goal
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a data extraction model"
            "only extract relevant data from the unstructured text"
            "if you don't know the value of what needs to be extracted"
            "return the value as null."
        ),
        ("human","{text}")
    ]
)

chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema = ManyPerson)
text = "down the road, comes a girl, long black hair, around 5 feet tall. walking next to her is her boyfriend Dave who has brown hair and 10 cm taller than her. "
response = chain.invoke(text)
print(response)




