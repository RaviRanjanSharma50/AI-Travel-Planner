import streamlit as st
import uuid

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from main import workflow, llm


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)


# --------------------------------
# SESSION INITIALIZATION
# --------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "trip_context" not in st.session_state:
    st.session_state.trip_context = ""

if "first_query" not in st.session_state:
    st.session_state.first_query = True


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("✈️ Travel Planner")

st.sidebar.write(
    "Your Agentic AI Travel Assistant"
)

if st.sidebar.button("➕ Start New Trip"):

    st.session_state.thread_id = str(uuid.uuid4())

    st.session_state.chat_history = []

    st.session_state.trip_context = ""

    st.session_state.first_query = True

    st.rerun()


st.sidebar.divider()

st.sidebar.write("### Technologies")

st.sidebar.write("🤖 LangGraph")
st.sidebar.write("🧠 Llama 3.2")
st.sidebar.write("🐘 PostgreSQL")
st.sidebar.write("🔎 Tavily Search")
st.sidebar.write("✈️ Flight API")


# --------------------------------
# MAIN HEADER
# --------------------------------

st.title("✈️ AI Travel Planner")

st.caption(
    "Your personal AI assistant for flights, hotels and itineraries."
)

st.divider()


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for chat in st.session_state.chat_history:

    with st.chat_message(chat["role"]):

        st.markdown(chat["content"])


# --------------------------------
# FOLLOW-UP QUESTION FUNCTION
# --------------------------------

def followup_response(user_query):

    conversation = ""

    for chat in st.session_state.chat_history:

        conversation += (
            f"{chat['role']}: {chat['content']}\n"
        )

    prompt = f"""
    You are an intelligent AI Travel Assistant.

    Previous Travel Context:
    {st.session_state.trip_context}

    Conversation History:
    {conversation}

    Latest User Question:
    {user_query}

    Instructions:
    1. Understand the previous travel conversation.
    2. Answer follow-up questions using previous context.
    3. Do not ask the user to repeat information already provided.
    4. If information is unavailable, clearly mention it.
    5. Provide clear and helpful answers.
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planning assistant."
        ),
        HumanMessage(content=prompt)
    ])

    return response.content


# --------------------------------
# CHAT INPUT
# --------------------------------

user_query = st.chat_input(
    "Ask me anything about your travel plans..."
)


if user_query:

    # Display user message

    with st.chat_message("user"):

        st.markdown(user_query)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })


    # --------------------------------
    # FIRST QUERY: EXECUTE LANGGRAPH
    # --------------------------------

    if st.session_state.first_query:

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        initial_state = {

            "messages": [
                HumanMessage(content=user_query)
            ],

            "user_query": user_query,

            "flight_results": "",

            "hotel_results": "",

            "itinerary": "",

            "llm_calls": 0
        }

        try:

            with st.spinner(
                "🤖 AI Agents are planning your journey..."
            ):

                result = workflow.invoke(
                    initial_state,
                    config=config
                )

                final_response = result["messages"][-1].content

                # Save travel context

                st.session_state.trip_context = f"""
                User Request:
                {user_query}

                Flight Results:
                {result.get('flight_results', '')}

                Hotel Results:
                {result.get('hotel_results', '')}

                Itinerary:
                {result.get('itinerary', '')}

                Final Recommendation:
                {final_response}
                """

                st.session_state.first_query = False

                st.session_state.llm_calls = result.get(
                    "llm_calls", 0
                )

        except Exception as e:

            final_response = f"Error: {str(e)}"


    # --------------------------------
    # FOLLOW-UP QUESTIONS
    # --------------------------------

    else:

        try:

            with st.spinner(
                "🤖 Understanding your follow-up question..."
            ):

                final_response = followup_response(
                    user_query
                )

        except Exception as e:

            final_response = f"Error: {str(e)}"


    # --------------------------------
    # DISPLAY ASSISTANT RESPONSE
    # --------------------------------

    with st.chat_message("assistant"):

        st.markdown(final_response)

    st.session_state.chat_history.append({

        "role": "assistant",

        "content": final_response

    })


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Powered by LangGraph | Ollama | PostgreSQL | Streamlit"
)