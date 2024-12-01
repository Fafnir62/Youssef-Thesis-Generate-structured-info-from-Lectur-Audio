import os
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

VECTORSTORE_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/vectorstore.faiss"
EXPORT_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/Export Station"
HISTORY_DIR = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/History"
HISTORY_FILE = os.path.join(HISTORY_DIR, "askLectures.json")

# Ensure the history directory and file exist
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)  # Initialize with an empty list


def load_history():
    """
    Load chat history from the JSON file. If the file is empty, return an empty list.
    """
    try:
        with open(HISTORY_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        return []


def save_to_history(question, answer):
    """
    Save a question-answer pair to the JSON file.
    """
    history = load_history()
    history.append({"question": question, "answer": answer})
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def generate_embeddings(transcript_path):
    """
    Generate embeddings for the given transcript and save them to FAISS.
    """
    with open(transcript_path, "r") as f:
        transcript = f.read()

    # Generate embeddings and overwrite FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts([transcript], embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore


def get_conversation_chain():
    """
    Create a conversational retrieval chain using LangChain.
    Ensure embeddings are regenerated for every new session.
    """
    transcript_path = os.path.join(EXPORT_PATH, "transcript.txt")
    if not os.path.exists(transcript_path):
        st.error("Transcript not found. Please generate the transcript first.")
        return None

    st.info("Generating embeddings for the transcript...")
    try:
        vectorstore = generate_embeddings(transcript_path)
        st.success("Embeddings generated successfully!")
    except Exception as e:
        st.error(f"Failed to generate embeddings: {e}")
        return None

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def app():
    """
    Streamlit app to chat with the course and save the history.
    """
    st.title("💬 Chat with Course")
    st.info("Chat with the transcript and get answers to your questions.")

    # Initialize conversation chain
    conversation_chain = get_conversation_chain()
    if conversation_chain is None:
        return

    # Input for user question
    user_question = st.text_input("Ask your question about the lectures:")

    if user_question:
        try:
            # Generate answer using embeddings
            answer = conversation_chain.run(user_question)

            # Display the interaction
            st.markdown(f"**You:** {user_question}")
            st.markdown(f"**Lecture:** {answer}")
            st.markdown("---")

            # Save the interaction to history
            save_to_history(user_question, answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

    # Display chat history under the input box
    history = load_history()
    if history:
        st.markdown("### Chat History")
        for interaction in reversed(history):
            st.markdown(f"**You:** {interaction['question']}")
            st.markdown(f"**Lecture:** {interaction['answer']}")
            st.markdown("---")
