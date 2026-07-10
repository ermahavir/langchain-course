from typing import List, Dict, Any

import streamlit as st

from backend.core import run_llm


def format_source(context_docs: List[str]) -> List[str]:
    # sources = []
    # for doc in (context_docs or []):
    #     meta = getattr(doc, "metadata", None) or {}
    #     source = meta.get("source", "Unknown source")
    #     sources.append(str(source))
    # return sources
    return [
        str(meta.get("source", "Unknown source"))
        for doc in (context_docs or [])
        if (meta := getattr(doc, "metadata", None) or {}) is not None
    ]

st.set_page_config(page_title="Langchain Doc Assistant", page_icon=":book:", layout="centered")
st.title("Langchain Doc Assistant")

with st.sidebar:
    st.subheader("Sessions")
    if st.button("Clear messages", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm a Langchain Doc Assistant. How can I help you today?",
            "sources": []
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}") # - displays source as a list item

prompt = st.chat_input("Ask a question about Langchain documentation")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving documentation..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")) or "(No answer available)"
                sources = format_source(result.get("context", []))

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        except Exception as e:
            st.error(f"Failed to generate response: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}", "sources": []})
