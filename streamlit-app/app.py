from collections.abc import Generator

import streamlit as st

from agents.agent_api import create_chatbot, stream_response


def response_generator() -> Generator[str, None, None]:
    """Generate a response from the chatbot word by word."""
    yield from stream_response(
        st.session_state.chatbot, st.session_state.messages[-1]["content"],
    )


st.title("Fake News Detector")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = create_chatbot()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    st.session_state.messages.append({"role": "assistant", "content": response})
