import os

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

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
# print(result)

#simple show responds
parser = StrOutputParser()
return_str = parser.invoke(result)
# print(return_str)

# define prompt template

prompt_template = ChatPromptTemplate.from_messages([
    ('system','please translate the following content into {language}'),
    ('user','{text}')
])

# create chains
chain = prompt_template | model | parser

# print(chain.invoke(msg))

print(chain.invoke({'language': 'spanish', 'text': 'learning Langchain is fun.'}))

# Deploy the program
# create fastAPI APP
app = FastAPI (title='test_app', version = 'V1.0', description = 'use langchain to translate into any language')

add_routes(
    app,
    chain,
    path = "/chain"
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host = "localhost", port = 8000)