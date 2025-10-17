import os
import re
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load .env (for consistent config even if unused here)
load_dotenv()

EXPORT_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/Export Station"

# Load the local model once
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

def clean_text(text):
    """Remove redundant whitespace, repeated words, or filler."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(Cloud computing\s*){2,}', 'Cloud computing ', text)
    return text.strip()

def generate_offline(prompt):
    """Call the offline model with a given prompt and return cleaned text."""
    result = generator(prompt)
    return clean_text(result[0]['generated_text'])

def process_transcript_offline():
    try:
        transcript_path = os.path.join(EXPORT_PATH, "transcript.txt")
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript not found in {EXPORT_PATH}")

        with open(transcript_path, "r") as f:
            transcript = f.read()

        # Split into smaller, overlapping chunks for better offline summarization
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        chunks = splitter.split_text(transcript)

        if not chunks:
            print("No usable chunks found.")
            return None

        structured_data = []
        seen_sections = set()

        for idx, chunk in enumerate(chunks):
            # Skip empty or very short chunks
            if len(chunk.strip().split()) < 20:
                continue

            print(f"[Offline] Processing chunk {idx + 1}/{len(chunks)}...")

            # Stronger, more directive prompts
            title_prompt = (
                "Write a clear, academic-style TITLE (max 10 words) for this lecture text:\n\n"
                f"{chunk}\n\nTITLE:"
            )
            summary_prompt = (
                "Write a CONCISE summary (2-3 sentences, avoid repeating phrases):\n\n"
                f"{chunk}\n\nSUMMARY:"
            )
            key_points_prompt = (
                "List 3-5 clear bullet-point KEY POINTS (short phrases):\n\n"
                f"{chunk}\n\nKEY POINTS:"
            )

            title = generate_offline(title_prompt)
            summary = generate_offline(summary_prompt)
            key_points = generate_offline(key_points_prompt)

            # Deduplicate sections
            unique_signature = (title.lower(), summary.lower())
            if unique_signature in seen_sections:
                continue
            seen_sections.add(unique_signature)

            # Append structured section
            section = {
                "title": title,
                "summary": summary,
                "key_points": key_points
            }
            structured_data.append(section)

        if not structured_data:
            print("Structured info generation resulted in no usable sections.")
            return None

        return structured_data

    except Exception as e:
        print(f"Error (offline structuring): {e}")
        return None
