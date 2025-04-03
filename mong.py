from pymongo import MongoClient
import whisper
import requests
from textblob import TextBlob
import nltk
nltk.download('punkt')

# MongoDB connection
client = MongoClient("mongodb+srv://Doyak123:doyak@123@doyak.kfpsy5k.mongodb.net/?retryWrites=true&w=majority&appName=Doyak")
db = client["Doyak"]
collection = db["videoCaptions"]

# Function to fetch video from cloud storage
def download_video(video_url, save_path="temp_video.mp4"):
    response = requests.get(video_url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    return save_path

# Function to generate captions using Whisper
def generate_caption(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["text"]

# Function to check caption authenticity using NLP
def check_authenticity(caption):
    analysis = TextBlob(caption)
    sentiment_score = analysis.sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)
    
    # Simple logic: If sentiment is too negative or caption is too short, flag it
    if sentiment_score < -0.5 or len(caption.split()) < 5:
        return False  # Caption is suspicious
    return True  # Caption is authentic

# Function to store data in MongoDB
def store_video_data(video_url, caption, is_authentic):
    video_data = {
        "video_url": video_url,
        "caption": caption,
        "authentic": is_authentic
    }
    collection.insert_one(video_data)
    print("Data stored successfully!")

# Example usage
video_url = ""  # Replace with actual cloud URL
video_path = download_video(video_url)
caption = generate_caption(video_path)
is_authentic = check_authenticity(caption)

if is_authentic:
    store_video_data(video_url, caption, True)
else:
    print("Caption failed authenticity check. Not storing in DB.")