from langgraph.graph import StateGraph ,START ,END
from typing import TypedDict , Annotated , Literal
from dotenv import load_dotenv
from pydantic import BaseModel , Field
from langchain_groq import ChatGroq
import operator
from langchain_core.messages import SystemMessage , HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0,
#     streaming=True
# )

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    streaming=True
)

class ChatState(TypedDict):

    messages : Annotated[list[BaseModel] , add_messages]

def chat_node(state: ChatState):

    # take user query from state
    messages = state['messages']

    # send to llm 
    response = llm.invoke(messages)

    # response store state
    return {'messages' :[response] }

checkpointer = InMemorySaver()
graph = StateGraph(ChatState)

# Add Node
graph.add_node('chat_node' , chat_node)

# Edges
graph.add_edge(START ,'chat_node')
graph.add_edge('chat_node' , END)

chatbot = graph.compile(checkpointer=checkpointer)