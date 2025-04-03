import whisper
import subprocess
import os
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Load Whisper model (for audio transcription)
model = whisper.load_model("large-v3")

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Sample pitch categories
CATEGORIES = ["AI", "Blockchain", "Agriculture", "FinTech", "FMCG", "Food Ventures", "Other"]

# Sample training data (expand this with real data)
training_data = {
    "AI": ["machine learning", "artificial intelligence", "deep learning", "neural networks", "computer vision", "natural language processing", "AI algorithms",
           "AI-powered solutions", "intelligent systems", "AI research", "cognitive computing", "AI ethics", "AI applications", "machine intelligence",
           "deep neural networks", "AI development", "AI platform", "AI strategy", "AI consulting", "AI transformation"],
    "Blockchain": ["crypto", "decentralized", "smart contracts", "NFT", "blockchain technology", "digital ledger", "cryptocurrency",
                   "blockchain solutions", "distributed ledger technology", "blockchain development", "crypto trading", "blockchain security",
                   "decentralized finance (DeFi)", "blockchain applications", "crypto investments", "blockchain consulting", "crypto assets",
                   "blockchain innovation", "digital currency", "crypto exchange"],
    "Agriculture": ["farming", "crops", "irrigation", "organic produce", "sustainable agriculture", "precision farming", "agribusiness",
                    "agricultural technology", "crop management", "livestock farming", "organic farming", "agricultural innovation",
                    "vertical farming", "agricultural solutions", "farm management software", "agricultural consulting", "crop science",
                    "agricultural research", "sustainable farming practices", "agricultural equipment"],
    "FinTech": ["banking", "payment gateways", "digital finance", "loans", "financial technology", "mobile payments", "online banking",
                "financial services", "fintech solutions", "digital payments", "financial innovation", "investment technology",
                "financial data analysis", "fintech startups", "financial software", "fintech consulting", "financial platforms",
                "digital banking solutions", "financial cybersecurity", "fintech industry"],
    "FMCG": ["consumer goods", "retail", "fast-moving", "packaged food", "household products", "personal care", "food and beverage",
             "consumer products", "retail industry", "FMCG sector", "packaged goods", "household essentials", "personal hygiene",
             "food processing", "FMCG brands", "retail management", "consumer behavior", "FMCG marketing", "supply chain management",
             "FMCG distribution"],
    "Food Ventures": ["restaurants", "food delivery", "cloud kitchen", "food tech", "culinary experiences", "sustainable food", "food innovation",
                      "food industry", "restaurant management", "online food ordering", "virtual restaurants", "food technology",
                      "gourmet food", "food startups", "food delivery platforms", "culinary arts", "food and beverage industry",
                      "food sustainability", "food trends", "food business"],
    "Other": ["general business", "miscellaneous", "startups", "entrepreneurship", "innovation", "technology", "business solutions",
              "business development", "business strategy", "startup ecosystem", "entrepreneurial ventures", "innovative solutions",
              "tech industry", "business consulting", "startup funding", "business management", "entrepreneurship development",
              "technology solutions", "business growth", "startup advice"]
}

# Prepare training data for NLP classification
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(
    [phrase for category in training_data.values() for phrase in category]
)
y_train = [category for category, phrases in training_data.items() for _ in phrases]

# Train a simple KNN classifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

def extract_audio(video_path, output_audio="extracted_audio.wav"):
    """Extracts audio from video using ffmpeg."""
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio} -y"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio

def transcribe_audio(audio_path):
    """Transcribes audio using Whisper AI."""
    result = model.transcribe(audio_path, language="en", fp16=False)
    text = " ".join([segment["text"] for segment in result["segments"]])
    return text

def classify_pitch(text):
    """Classifies the pitch topic using NLP."""
    text_vectorized = vectorizer.transform([text])
    category = classifier.predict(text_vectorized)[0]
    return category

def is_abusive(text, threshold=0.8):
    """Checks if the text is abusive using sentiment analysis."""
    result = sentiment_pipeline(text)[0]
    # print(result)
    # If the 'label' is 'NEGATIVE' and the score is above the threshold, consider it abusive
    if result['label'] == 'NEGATIVE' and result['score'] > threshold:
        return True
    return False

def generate_captions(video_path, output_srt="captionsnlp.srt"):
    """Generates captions and classifies the pitch topic."""
    audio_file = extract_audio(video_path)
    transcript = transcribe_audio(audio_file)
    
    if is_abusive(transcript):
        print("⚠️ Abusive language detected. Captions will not be generated.")
        return  # Stop the function if abusive language is detected
    
    pitch_category = classify_pitch(transcript)

    with open(output_srt, "w") as f:
        f.write(f"1\n00:00:00,000 --> 00:10:00,000\n{transcript}\n\n")
    
    print(f"✅ Captions saved as {output_srt}")
    print(f"🎯 Pitch Category Identified: {pitch_category}")

# Run the script
video_file = "video2.mp4"  # Change this to your actual video file
generate_captions(video_file)
