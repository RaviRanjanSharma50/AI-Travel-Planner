import streamlit as st
import uuid

from langchain_core.messages import HumanMessage
from main import workflow


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)


# --------------------------------
# Session Management
# --------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "travel_result" not in st.session_state:
    st.session_state.travel_result = None


# --------------------------------
# Header
# --------------------------------

st.title("✈️ AI Travel Planner")

st.markdown(
    """
    ### Your Personal Agentic AI Travel Assistant

    Plan your journey using multiple AI agents:
    - ✈️ Flight Agent
    - 🏨 Hotel Agent
    - 🗺️ Itinerary Agent
    - 🤖 Final Response Agent
    """
)

st.divider()


# --------------------------------
# User Input
# --------------------------------

st.subheader("🌍 Plan Your Trip")

with st.form("travel_form"):

    user_input = st.text_area(
        "Where would you like to travel?",
        placeholder="Example: Plan a 5-day trip from Delhi to Goa "
                    "with budget-friendly hotels.",
        height=120
    )

    submit = st.form_submit_button(
        "🚀 Generate Travel Plan",
        use_container_width=True
    )


# --------------------------------
# Execute LangGraph Workflow
# --------------------------------

if submit:

    if not user_input.strip():

        st.warning("Please enter your travel requirements.")

    else:

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        initial_state = {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
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

                st.session_state.travel_result = result

        except Exception as e:

            st.error(f"Something went wrong: {str(e)}")


# --------------------------------
# Display Results
# --------------------------------

if st.session_state.travel_result:

    result = st.session_state.travel_result

    st.divider()

    st.header("🎯 Your Travel Plan")

    # Flight Results
    with st.expander("✈️ Flight Information", expanded=True):

        st.write(result.get("flight_results", "No flight data available"))


    # Hotel Results
    with st.expander("🏨 Hotel Information", expanded=True):

        st.write(result.get("hotel_results", "No hotel data available"))


    # Itinerary
    with st.expander("🗺️ Travel Itinerary", expanded=True):

        st.markdown(result.get("itinerary", "No itinerary available"))


    # Final Response
    st.divider()

    st.subheader("🤖 Final AI Recommendation")

    messages = result.get("messages", [])

    if messages:

        st.markdown(messages[-1].content)

    # LLM Usage
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "LLM / Agent Calls",
            result.get("llm_calls", 0)
        )

    with col2:
        st.metric(
            "Workflow Status",
            "Completed"
        )