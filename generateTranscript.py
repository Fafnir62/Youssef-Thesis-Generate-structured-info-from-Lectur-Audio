import os
import whisper

# Define the export path for saving the transcript
EXPORT_PATH = "/home/fafnir/Alpha/_Python/Python Current/Youssef Thesis/Export Station"

def transcribe_audio(file_path):

    try:
        # Check if the Whisper library is properly installed
        if not hasattr(whisper, "load_model"):
            raise AttributeError("The Whisper library does not have 'load_model'. Ensure openai-whisper is installed.")

        # Load the Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Transcribing audio...")

        # Perform the transcription
        result = model.transcribe(file_path)
        transcript = result["text"]

        # Ensure the export directory exists
        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)

        # Save the transcript as a text file
        transcript_path = os.path.join(EXPORT_PATH, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write(transcript)

        print(f"Transcript saved to {transcript_path}")
        return transcript_path, transcript

    except AttributeError as e:
        print(f"Error: {e}")
        print("Please ensure the correct version of Whisper is installed using 'pip install -U openai-whisper'.")
        return None, None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None, None
