import os
import json
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

VECTORSTORE_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/vectorstore_offline.faiss"
EXPORT_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/Export Station"
HISTORY_DIR = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/History"
HISTORY_FILE = os.path.join(HISTORY_DIR, "askLecturesOffline.json")

os.makedirs(HISTORY_DIR, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

def load_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        return []

def save_to_history(question, answer):
    history = load_history()
    history.append({"question": question, "answer": answer})
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def generate_offline_embeddings(transcript_path):
    with open(transcript_path, "r") as f:
        transcript = f.read()

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts([transcript], embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def get_conversation_chain_offline():
    transcript_path = os.path.join(EXPORT_PATH, "transcript.txt")
    if not os.path.exists(transcript_path):
        st.error("Transcript not found. Please generate the transcript first.")
        return None

    st.info("Generating offline embeddings...")
    try:
        vectorstore = generate_offline_embeddings(transcript_path)
        st.success("Embeddings generated successfully!")
    except Exception as e:
        st.error(f"Failed to generate embeddings: {e}")
        return None

    # Load Flan-T5 for offline generation
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def app():
    st.title("💬 Offline Chat with Course")
    st.info("Chat with the transcript using fully offline models (no internet).")

    conversation_chain = get_conversation_chain_offline()
    if conversation_chain is None:
        return

    user_question = st.text_input("Ask your question about the lectures:")

    if user_question:
        try:
            answer = conversation_chain.run(user_question)
            st.markdown(f"**You:** {user_question}")
            st.markdown(f"**Lecture:** {answer}")
            st.markdown("---")
            save_to_history(user_question, answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

    history = load_history()
    if history:
        st.markdown("### Chat History")
        for interaction in reversed(history):
            st.markdown(f"**You:** {interaction['question']}")
            st.markdown(f"**Lecture:** {interaction['answer']}")
            st.markdown("---")
