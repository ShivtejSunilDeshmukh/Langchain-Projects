# chat_ui.py
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Initialize model
model = ChatOpenAI(
    model="openrouter/free",  # Replace "openrouter/free" with a valid model
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize session state for conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Simple LangChain ChatBot")

# User input
user_input = st.text_input("You:")

if user_input:
    # Append human message
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Get response from model
    response = model.invoke(st.session_state.messages)

    # Append AI message
    st.session_state.messages.append(AIMessage(content=response.content))

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Bot:** {msg.content}")