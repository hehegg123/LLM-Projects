#  Using chain and agents to achieve data retrieval from SQL and provide natraul language
# 1. use model to change user input to SQL inspection
# 2, execute SQL search
# 3. Use model to respond to user

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase



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

print(db.get_usable_table_names())

# directly pull answers from SQL using LLM Model
test_chain = create_sql_query_chain(model, db)
response = test_chain.invoke({'question': 'how many rows are there in the actor table?'})
print(response)

# strip the SQLQuery part of the function
def strip_sql_prefix(sql_query_with_prefix: str) -> str:
    if sql_query_with_prefix.startswith("SQLQuery: "):
        return sql_query_with_prefix[len("SQLQuery: "):].strip()
    return sql_query_with_prefix.strip()

strip_prefix_runnable = RunnableLambda(strip_sql_prefix)

answer_prompt = ChatPromptTemplate.from_template(
    """given the following SQL database, SQL command and the user question, provide the full sentence answer to the question:
    Question: {question}
    SQL Query:{query}
    SQL Result:{result}
    Answer:  
    """
)
# establish the tool to execute sql command
execute_sql_tool = QuerySQLDatabaseTool(db=db)

# create chain to generate SQL query and execute SQL query
chain = (RunnablePassthrough.assign(query = test_chain | strip_prefix_runnable).assign(result = itemgetter('query') | execute_sql_tool)
        | answer_prompt
        | model
        | StrOutputParser()
         )

response1 = chain.invoke(input = {'question': 'how many rows are there in the actor table?'})
print(response1)