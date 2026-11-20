
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #60A5FA;
}
.subtitle {
    text-align: center;
    color: #CBD5E1;
}
</style>
""", unsafe_allow_html=True)

# Application heading
st.markdown(
    '<div class="title">🤖 Advanced RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about your documents</div>',
    unsafe_allow_html=True
)

st.divider()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("📚 Document Assistant")

    st.write("**Knowledge Base:**")
    st.write("Company Policy PDF")

    st.divider()

    st.subheader("Conversation")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.info("Powered by LangChain, FAISS and Groq")

# Display previous messages
for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "assistant"

    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
question = st.chat_input("Ask something about your document...")

if question:

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    # Store user message
    st.session_state.chat_history.append(
        HumanMessage(content=question)
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                answer = rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history[-11:-1]
                })

                st.markdown(answer)

                # Store AI response
                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
