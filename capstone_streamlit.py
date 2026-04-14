import streamlit as st
import uuid

from agent import CapstoneState, build_app

st.set_page_config(
    page_title="Agentic AI Course Assistant",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Part 7: Cache expensive resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_app():
    return build_app()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Course Assistant")
    st.markdown("**Domain:** Agentic AI (13-day course)")
    st.markdown("**User:** B.Tech 4th year students")
    st.markdown("---")
    st.markdown("**Topics covered:**")
    topics = [
        "Day 1 — LLM APIs & First Agent",
        "Day 2 — Tool Calling",
        "Day 3 — Memory Systems",
        "Day 4 — RAG Foundations",
        "Day 5 — LangChain Deep Dive",
        "Day 6 — CrewAI Multi-Agent",
        "Day 7 — Advanced CrewAI",
        "Day 8 — LangGraph Workflows",
        "Day 9 — Self-Reflection",
        "Day 10 — RAG + Memory",
        "Day 11 — RAGAS Evaluation",
        "Day 12 — Deployment",
        "Day 13 — Capstone",
    ]
    for t in topics:
        st.markdown(f"- {t}")

    st.markdown("---")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph + Groq (Llama 3.3)")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Agentic AI Course Assistant")
st.caption("Ask me anything about the 13-day Agentic AI course.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))

# Chat input
if prompt := st.chat_input("Ask about any session, concept, or topic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state: CapstoneState = {
                "question": prompt,
                "route": "",
                "retrieved": "",
                "sources": [],
                "tool_result": "",
                "answer": "",
                "faithfulness": 1.0,
                "eval_retries": 0,
            }
            result = get_app().invoke(initial_state, config=config)

        answer = result["answer"]
        sources = result["sources"]
        route = result["route"]
        faith = result["faithfulness"]

        st.markdown(answer)
        if sources:
            st.caption("Sources: " + " · ".join(sources))

        with st.expander("Debug info", expanded=False):
            st.json({
                "route": route,
                "faithfulness": round(faith, 3),
                "thread_id": st.session_state.thread_id,
                "eval_retries": result["eval_retries"],
            })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
