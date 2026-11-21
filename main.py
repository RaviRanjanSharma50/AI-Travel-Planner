import os
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (AnyMessage, HumanMessage, AIMessage, SystemMessage)
from langchain_ollama import ChatOllama

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(model='llama3.2', temperature=0)
DATABASE_URL = os.getenv('DATABASE_URL')

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

def flight_agent(state: TravelState):
    query = state['user_query']
    flight_data = search_flights(query)
    return {
        'flight_results': flight_data,
        'messages':[
            AIMessage(content=f'Flight result fetched')
        ],
        'llm_calls':state.get('llm_calls', 0) + 1
    }


def hotel_agent(state: TravelState):
    query = f"best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        'hotel_results': hotel_results,
        'messages': [
            AIMessage(content="Hotel information fetched")
        ],
        'llm_calls': state.get('llm_calls', 0) + 1
    }



def itinerary_agent(state: TravelState):
    prompt = f"""
        Create a travel itenerary.
        User Query:
        {state['user_query']}

        Flight Results:
        {state['flight_results']}

        Hotel Results:
        {state['hotel_results']}
        """
    response = llm.invoke([
        SystemMessage(
            content="You are a expert Travel planner"
        ),
        HumanMessage(content=prompt)
    ])    

    return {
        'itinerary': response.content,
        'messages':[response],
        'llm_calls': state.get('llm_calls', 0) + 1
    }


def final_agent(state: TravelState):

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke([
        HumanMessage(content= final_prompt)
    ])

    return {
        'messages': [response],
        'llm_calls': state.get('llm_calls', 0) + 1
    }


graph = StateGraph(TravelState)

graph.add_node('flight_agent', flight_agent)
graph.add_node('hotel_agent', hotel_agent)
graph.add_node('itinerary_agent', itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, 'flight_agent')
graph.add_edge('flight_agent', 'hotel_agent')
graph.add_edge('hotel_agent', 'itinerary_agent')
graph.add_edge('itinerary_agent','final_agent')
graph.add_edge('final_agent', END)

_conn = psycopg.connect(DATABASE_URL)
checkpointer = PostgresSaver(_conn)
checkpointer.setup() 

workflow = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config ={
        'configurable': {
            'thread_id': 'user_ravi'
        }

    }

    user_input = input("Enter travel request: ")

    result = workflow.invoke(
        {
            'messages': [
                HumanMessage(content= user_input)
            ],
            'user_query': user_input,
            'flight_results': "",
            'hotel_results':"",
            'itinerary': "",
            'llm_calls': 0
        },
        config=config
    )


    print("\n FINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)