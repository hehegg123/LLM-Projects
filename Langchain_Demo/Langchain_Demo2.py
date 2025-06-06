from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
import os


os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# --- CORRECTED PROMPT TEMPLATE ---
# Use two MessagesPlaceholder: one for 'chat_history' and one for the 'my_msg' (current input)
prompt_template = ChatPromptTemplate.from_messages([  # defines the structured input given to the LLM
    SystemMessage(content='you are a helpful bot, always answer question in {language}'),
    #MessagesPlaceholder(variable_name='chat_history'), # Placeholder for the conversation history
    MessagesPlaceholder(variable_name='my_msg')        # Placeholder for the current user's message(s)
])

Chain = prompt_template | model

# store chat history
store = {} # a Dictionary to store all user's chat history. Use session ID as key


# Function to use to receive a session_id and return a chat
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- CORRECTED RunnableWithMessageHistory ---
# Explicitly define history_messages_key to link it to the 'chat_history' placeholder in the prompt
do_message = RunnableWithMessageHistory(
    Chain,
    get_session_history,
    input_messages_key = 'my_msg',       # The key from the invoke input that contains the current messages
    #history_messages_key = 'chat_history' # The variable name in the prompt template for the history
)

#define config
config = {'configurable': {'session_id':'hehegg123'}} # define a session for the current chat

# First Chat
print("--- First Chat ---")
response1 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content = 'Hi, This is Yiwei')], # Current user message
        'language': 'English'
    },
    config = config
)

print(f"Bot: {response1.content}")

# Second Chat
print("\n--- Second Chat ---")
response2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content = 'what is my name?')], # Current user message
        'language': 'english'
    },
    config = config
)

print(f"Bot: {response2.content}")

config2 = {'configurable': {'session_id':'hehegg321'}} # define a session for the current chat
# Third round
print("--- Third Chat ---")
for response in do_message.stream(
    {
        'my_msg': [HumanMessage(content = 'what is a joke?')], # Current user message
        'language': 'mandarin'
    },
    config = config2
):
    print(response.content, end = '-')