import streamlit as st
import re
from harry_logic import ask_harry 

st.set_page_config(page_title="The Gryffindor Common Room", layout="centered")
st.title("⚡ Ask Harry Potter")
st.caption("The Boy Who Lived")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_list" not in st.session_state:
    st.session_state.history_list = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Speak to Harry..."):

    # 1. Add User Message to UI State
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Harry is thinking..."):
            response_data = ask_harry(prompt, st.session_state.history_list)
        response_text = response_data["response"]
        reasoning = response_data.get("reasoning", {})
        sources = response_data.get("sources", [])

        with st.sidebar:
            st.subheader("🧙‍♂️ Persona Internal State")
            st.write(f"**Motive:** {reasoning.get('motive', 'N/A')}")
            st.write(f"**Conflict:** {reasoning.get('internal_conflict', 'N/A')}")
            st.info(f"**Reasoning Trace:** {reasoning.get('reasoning_trace', 'N/A')}")

        st.markdown(response_text)

        if sources:
            with st.expander("Lore References (Hogwarts Archives)"):
                for source in sources:
                    st.caption(f"📖 {source}")
    st.session_state.messages.append({"role": "assistant", "content": response_text})