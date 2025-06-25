from langchain.agents import AgentExecutor
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import chat_agent_executor



os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

#sqlalchemy
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'sakila'
USERNAME = 'root'
PASSWORD = '12345'

MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db= SQLDatabase.from_uri(MYSQL_URI)

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

#use an agent to complete the consolidation of the databases
system_prompt = """
You are an agent designed to interact with a SQL database for processing.
Given a user input problem, create a syntactically correct SQL statement to execute, then check the query results and return the answer based on the results.

Unless the user specifies the number of examples they want to get, the SQL query will always be limited to a maximum of 10 results.
You can sort the results by relevance to return the best matching data in the MySQL database.
You can use tools that interact with the database. You must check the query carefully before executing it. If an error occurs during execution, please rewrite the query and try again.
Do not make any DML statements (insert, update, delete, etc.) to the database.

First, you should look at the tables in the database to see what you can query.
Do not skip this step.
Then query the schema of the most relevant tables.
"""


system_message = SystemMessage(content=system_prompt)

#create agent
agent_executor = chat_agent_executor.create_tool_calling_executor(model,tools)

response = agent_executor.invoke({'messages': [SystemMessage(content=system_prompt), HumanMessage(content='how many rows are there in the actor table?')]},
                                 config={"recursion_limit": 50})
result = response['messages']
print(result)
print(result[len(result)-1])
