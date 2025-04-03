import whisper
import subprocess
import spacy
from collections import Counter

# Load Whisper model for transcription
model = whisper.load_model("large-v3")

# Load NLP model for topic classification
nlp = spacy.load("en_core_web_sm")

# Predefined topic keywords
TOPIC_KEYWORDS = {
    "AI": ["machine learning", "deep learning", "artificial intelligence", "neural networks", "automation"],
    "Blockchain": ["cryptocurrency", "smart contract", "decentralized", "ledger", "bitcoin"],
    "Agriculture": ["farming", "crops", "harvest", "soil", "organic"],
    "FinTech": ["banking", "finance", "investment", "trading", "digital payment"],
    "FMCG": ["consumer goods", "retail", "fast-moving", "supply chain"]
}

def extract_audio(video_path, output_audio="audio.wav"):
    """Extracts audio from video using ffmpeg."""
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio} -y"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio

def transcribe_audio(audio_path):
    """Transcribes audio using Whisper AI."""
    result = model.transcribe(audio_path, language="en", fp16=False)
    transcript = " ".join([segment["text"] for segment in result["segments"]])
    return transcript

def classify_pitch(text):
    """Classifies the pitch based on NLP keyword matching."""
    doc = nlp(text.lower())
    word_freq = Counter([token.text for token in doc])

    # Check frequency of topic-related words
    topic_scores = {topic: sum(word_freq[word] for word in words if word in word_freq) for topic, words in TOPIC_KEYWORDS.items()}
    
    # Get the highest-scoring topic
    best_match = max(topic_scores, key=topic_scores.get)
    return best_match if topic_scores[best_match] > 0 else "Unknown"

def generate_captions(video_path, output_srt="captions.srt"):
    """Generates captions in paragraph format and classifies the pitch topic."""
    audio_file = extract_audio(video_path)
    transcript = transcribe_audio(audio_file)
    topic = classify_pitch(transcript)

    with open(output_srt, "w") as f:
        f.write(f"1\n00:00:00,000 --> 00:00:30,000\n{transcript}\n\n")

    print(f"✅ Captions saved as {output_srt}")
    print(f"📌 Pitch Category: {topic}")

# Run the script
video_file = "video.mp4"  # Change this to your actual video file
generate_captions(video_file)
