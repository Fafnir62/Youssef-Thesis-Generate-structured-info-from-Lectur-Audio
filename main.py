import streamlit as st
from streamlit_option_menu import option_menu
from generateTranscript import transcribe_audio
from structuredInfo import process_transcript
from relatedArticles import get_related_articles
from chatCourse import app as chat_course_app
import os
import shutil

EXPORT_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/Export Station"

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func, icon):
        self.apps.append({
            "title": title,
            "function": func,
            "icon": icon
        })

    def run(self):
        with st.sidebar:
            selected_app = option_menu(
                menu_title="Lecture Tools",
                options=[app["title"] for app in self.apps],
                icons=[app["icon"] for app in self.apps],
                menu_icon="book",
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                    "icon": {"color": "#0099ff", "font-size": "23px"},
                    "nav-link": {"font-size": "18px", "text-align": "left", "--hover-color": "#e8efff"},
                    "nav-link-selected": {"background-color": "#003d99", "color": "white"},
                }
            )

        for app in self.apps:
            if app["title"] == selected_app:
                app["function"]()


def transcript_page():
    st.title("🎤 Transcribe MP3 Lecture")
    st.info("Upload an MP3 file and generate its transcript.")

    uploaded_file = st.file_uploader("Upload MP3 Lecture", type=["mp3"])

    if uploaded_file:
        file_path = os.path.join(EXPORT_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"File uploaded: {file_path}")

        if st.button("Generate Transcript"):
            transcript_path, transcript = transcribe_audio(file_path)
            if transcript:
                st.success(f"Transcript generated successfully! File saved at: {transcript_path}")
                st.text_area("Transcript", transcript, height=300)
            else:
                st.error("Failed to generate transcript.")

    if st.button("Reset"):
        if os.path.exists(EXPORT_PATH):
            shutil.rmtree(EXPORT_PATH)
            os.makedirs(EXPORT_PATH)
        st.success("All files in Export Station have been deleted.")


def structured_info_page():
    st.title("Generate Structured Information")
    st.info("Organize lecture transcript into structured sections with titles, summaries, and key points.")

    if st.button("Generate Structured Information"):
        structured_data = process_transcript()
        if structured_data:
            st.success("Structured information generated successfully!")
            for idx, section in enumerate(structured_data):
                st.markdown(f"### {section['title']}")
                st.markdown(f"{section['summary']}")
                if "key_points" in section:
                    st.markdown("## **Key Points:**")
                    st.markdown(f"{section['key_points']}")
        else:
            st.error("Failed to generate structured information. Ensure a transcript is available.")


def related_articles_page():
    st.title("Related Articles")
    st.info("Find articles related to the topics discussed in the transcript.")

    if st.button("Find Related Articles"):
        articles = get_related_articles()
        if articles:
            st.success("Related articles retrieved successfully!")
            for article in articles:
                st.markdown(f"### [{article['title']}]({article['link']})")
                st.markdown(f"{article['description']}")
        else:
            st.error("No articles found. Ensure a transcript is available and try again.")


def chat_course_page():
    st.title("Chat with Course")
    st.info("Chat with the transcript and get answers to your questions. Explore detailed explanations.")

    # The chatCourse.py app function handles the functionality
    chat_course_app()


if __name__ == "__main__":
    st.info("Initializing the application... Please wait.")
    app = MultiApp()
    app.add_app("🎤 Transcribe Lecture", transcript_page, "microphone")
    app.add_app("tructured Information", structured_info_page, "file-text")
    app.add_app("Related Articles", related_articles_page, "link")
    app.add_app("Chat with Course", chat_course_page, "chat")
    app.run()
