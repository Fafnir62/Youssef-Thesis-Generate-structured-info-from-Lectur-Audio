import os
import requests
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI

# Load environment variables
load_dotenv()

EXPORT_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/Export Station"
SERP_API_KEY = os.getenv("SERP_API_KEY")
SERP_API_URL = "https://serpapi.com/search"


def extract_academic_keywords(transcript, num_keywords=7):
    """
    Extract academic keywords from the transcript in chunks to avoid exceeding token limits.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the .env file.")

    # Initialize the OpenAI model
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

    # Define the prompt template
    prompt_template = PromptTemplate(
        template=(
            "Extract {num_keywords} academic and research-oriented keywords from this text:\n\n{text}\n\n"
            "Keywords:"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Chunk the transcript into smaller parts if too long
    max_chunk_size = 1000  # Approximate size for safe token count
    chunks = [transcript[i:i + max_chunk_size] for i in range(0, len(transcript), max_chunk_size)]

    keywords = set()
    for chunk in chunks:
        try:
            chunk_keywords = chain.run({"text": chunk, "num_keywords": num_keywords}).strip()
            keywords.update(chunk_keywords.split(", "))
        except Exception as e:
            print(f"Error during keyword extraction for a chunk: {e}")

    return list(keywords)[:num_keywords]  # Return top `num_keywords`


def retrieve_articles_online(query, num_results=5):
    """
    Retrieve related articles using the SERP API based on a search query in Google Scholar.
    """
    if not SERP_API_KEY:
        raise ValueError("SERP_API_KEY is not set in the .env file.")

    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results,
        "engine": "google_scholar",
    }

    response = requests.get(SERP_API_URL, params=params)
    response.raise_for_status()
    search_results = response.json()

    articles = []
    for result in search_results.get("organic_results", []):
        articles.append({
            "title": result.get("title"),
            "link": result.get("link"),
            "description": result.get("snippet", "No description available."),
        })

    return articles


def get_related_articles():
    """
    Process the transcript, extract academic keywords, and retrieve related articles.
    """
    try:
        transcript_path = os.path.join(EXPORT_PATH, "transcript.txt")
        if not os.path.exists(transcript_path):
            raise FileNotFoundError("Transcript file not found. Please generate the transcript first.")

        with open(transcript_path, "r") as f:
            transcript = f.read()

        # Extract academic keywords
        keywords = extract_academic_keywords(transcript)
        print(f"Generated Academic Keywords: {keywords}")

        # Join keywords for the search query
        query = ", ".join(keywords)
        print(f"Query Sent for Article Retrieval: {query}")

        # Retrieve articles online using the keywords
        articles = retrieve_articles_online(query)
        return articles

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    except Exception as e:
        print(f"Error during article retrieval: {e}")
        return []
