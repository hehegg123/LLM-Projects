import os
import getpass
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from typing import TypedDict, Annotated, List, Dict, Any
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

class Garment:
    name: str
    kind: str  # "top", "bottom", "outer", "shoes", "accessory"
    warmth: int  # set from 1-5
    waterproof: bool = False
    formal: bool = False
    Color: str = "neutral"

Wardrobe = List[Garment]

# Example wardrobe
WARDROBE: Wardrobe = [
    Garment("cotton t-shirt", "top", 1, color="white"),
    Garment("long-sleeve tee", "top", 2, color="navy"),
    Garment("button-down shirt", "top", 2, formal=True, color="light blue"),
    Garment("hoodie", "outer", 3, color="grey"),
    Garment("light jacket", "outer", 3, waterproof=False, color="olive"),
    Garment("rain shell", "outer", 2, waterproof=True, color="black"),
    Garment("wool coat", "outer", 5, color="charcoal"),
    Garment("chinos", "bottom", 2, formal=True, color="khaki"),
    Garment("jeans", "bottom", 2, color="indigo"),
    Garment("thermal leggings", "bottom", 3, color="black"),
    Garment("sneakers", "shoes", 1, color="white"),
    Garment("waterproof boots", "shoes", 3, waterproof=True, color="brown"),
    Garment("beanie", "accessory", 2, color="black"),
    Garment("scarf", "accessory", 2, color="grey"),
    Garment("umbrella", "accessory", 0, waterproof=True, color="black"),
]

class State(TypedDict):
    messages: Annotated[list,add_messages]