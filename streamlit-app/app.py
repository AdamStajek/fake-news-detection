from collections.abc import Generator

import streamlit as st

from agents.agent_api import create_chatbot, stream_response
from agents.chatbot.tools import get_available_tools


def response_generator() -> Generator[str, None, None]:
    """Generate a response from the chatbot word by word."""
    yield from stream_response(
        st.session_state.chatbot, st.session_state.messages[-1]["content"],
    )


st.title("Fake News Detector")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.subheader("Configuration")

    model = st.selectbox(
        "Select Language Model:",
        ["Gemini 2.5 Flash", "Gemini 2.5 Pro"],
    )

    st.write("Select Tools (excluding vectorstore):")
    available_tools = get_available_tools()

    tool_labels = {
        "web_search": "Web Search (DuckDuckGo)",
        "verify_claim_sources": "Verify Claim Sources",
        "search_research_papers": "Search Research Papers (arXiv)",
        "analyze_news_source": "Analyze News Source Credibility",
    }

    selected_tools = []
    for tool_key, tool_label in tool_labels.items():
        if st.checkbox(tool_label, value=True, key=tool_key):
            selected_tools.append(tool_key)

    if st.button("Start Chat"):
        if not selected_tools:
            st.warning("Please select at least one tool to continue.")
        else:
            st.session_state.model = model
            st.session_state.selected_tools = selected_tools
            st.session_state.chatbot = create_chatbot(
                "agent",
                model,
                schema=None,
                vectorstore_collection_name="overall_covid_papers",
                selected_tools=selected_tools,
            )
            st.rerun()
else:
    st.sidebar.subheader("Current Configuration")
    st.sidebar.write(f"**Model:** {st.session_state.model}")
    st.sidebar.write("**Active Tools:**")
    tool_labels = {
        "web_search": "Web Search",
        "verify_claim_sources": "Verify Claim Sources",
        "search_research_papers": "Search Research Papers",
        "analyze_news_source": "Analyze News Source",
    }
    for tool in st.session_state.selected_tools:
        st.sidebar.write(f"- {tool_labels.get(tool, tool)}")

    if st.sidebar.button("Reset Configuration"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about any news or claims"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
        st.session_state.messages.append({"role": "assistant", "content": response})
