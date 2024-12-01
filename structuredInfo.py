import os
import random
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EXPORT_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/Export Station"

def process_transcript():
    """
    Load the transcript from the Export Station and process it using LangChain.
    Each chunk gets a title and a summary, and key points are added at random intervals.

    Returns:
        list: A list of dictionaries containing structured sections.
    """
    try:
        # Load the OpenAI API key from the environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set it in the .env file.")

        # Check if the transcript exists in Export Station
        transcript_path = os.path.join(EXPORT_PATH, "transcript.txt")
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found in {EXPORT_PATH}. Please generate it first.")

        # Read the transcript
        with open(transcript_path, "r") as f:
            transcript = f.read()

        # Initialize LangChain components
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

        # Prompt templates
        title_prompt = PromptTemplate(template="Provide a concise title for the following text:\n\n{text}\n")
        summary_prompt = PromptTemplate(template="Summarize the following text:\n\n{text}\n")
        key_points_prompt = PromptTemplate(template="Extract key points from the following text:\n\n{text}\n")

        title_chain = LLMChain(llm=llm, prompt=title_prompt)
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        key_points_chain = LLMChain(llm=llm, prompt=key_points_prompt)

        # Split and process transcript
        chunks = splitter.split_text(transcript)
        structured_data = []

        # Initialize random interval for key points
        key_points_interval = random.randint(1, 5)
        next_key_points_chunk = key_points_interval

        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}...")

            title = title_chain.run({"text": chunk}).strip()
            summary = summary_chain.run({"text": chunk}).strip()

            section = {
                "title": title,
                "summary": summary,
            }

            # Add key points at random intervals
            if idx + 1 == next_key_points_chunk:
                key_points = key_points_chain.run({"text": chunk}).strip()
                section["key_points"] = key_points

                # Set the next interval
                key_points_interval = random.randint(1, 5)
                next_key_points_chunk = idx + 1 + key_points_interval

            structured_data.append(section)

        return structured_data

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error during structuring: {e}")
        return None
