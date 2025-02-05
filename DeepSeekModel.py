import streamlit as st
from langchain_community.llms import Ollama

st.title("LLM Based Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Post your question here.")
if user_input:
    st.chat_message("user").write(user_input)

    llm = Ollama(model="deepseek-r1:1.5b")
    response = llm.invoke(user_input)

    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    st.chat_message("assistant").write(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

#streamlit run DeepSeekModel.py
