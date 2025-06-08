
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
import os

from langgraph.prebuilt import chat_agent_executor

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

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
tools = [search]
Agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
response = Agent_executor.invoke({'messages':[HumanMessage(content="where is the capital of US?")]})
print(response['messages'])
response2 = Agent_executor.invoke({'messages':[HumanMessage(content="how's the weather in washington D.C.?")]})
print(response2['messages'])
print(response2['messages'][2].content)
print(response2['messages'][-1].content)
