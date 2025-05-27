import os

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = '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'

#pull the model

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

#prompt

msg = [
    SystemMessage(content = 'translate the following content into Spanish'),
    HumanMessage(content = 'hello, where are you heading?')
]

result = model.invoke(msg)
print(result)

#simple show responds
parser = StrOutputParser()
return_str = parser.invoke(result)
print(return_str)

# create chains
chain = model | parser
print(chain.invoke(msg))