import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

st.title("BRiX Chatbot")

with open("data.json", "r") as f:
    knowledge_base = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
kb_texts = [item["message"] for item in knowledge_base]
kb_embeddings = model.encode(kb_texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

def retrieve_knowledge(question):
    question_embedding = model.encode([question], convert_to_numpy=True)
    _, idx = index.search(question_embedding, 1)
    return knowledge_base[idx[0][0]]["response"] if idx[0][0] >= 0 else "Sorry, I don't have any information on that."

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Post your question here.")
if user_input:
    st.chat_message("user").write(user_input)

    retrieved_info = retrieve_knowledge(user_input)
    prompt = f"Based on the following relevant information, provide a precise and context-aware answer : {retrieved_info} \n\nUser Query: {user_input} \nAnswer:"

    llm = Ollama(model="deepseek-r1:1.5b")
    response = llm.invoke(prompt)

    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    st.chat_message("assistant").write(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

#streamlit run Brix_Chatbot.py
