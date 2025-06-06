
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
import os

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# without agent
# result = model.invoke([HumanMessage(content = 'what is current weather at West Lafayette, IN?')])
# print(result)

#use Tavily as search engine
search = TavilySearchResults(Max_results = 3) # only return 3 results.
model.bind_tools([search])

#let's bind the tool with model
model_with_tools = model.bind_tools([search])
# let the model decide if it needs the tool

# response = model_with_tools.invoke([HumanMessage(content="use tavily to tell me what is the weather like at west lafayette IN?")])
#print(f'Model_Result_Content:{response.content}')
#print(f'Tools_result_content:{response.tool_calls}')

# creating an agent


