# import whisper
# import cv2
# import numpy as np
# import subprocess  # For using ffmpeg to extract audio
# from deepface import DeepFace
# import os

# # Load Whisper model (use "large-v3" for best accuracy)
# model = whisper.load_model("large-v3")

# def extract_audio(video_path, output_audio="audio.wav"):
#     """Extracts audio from video using ffmpeg."""
#     command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio} -y"
#     subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     return output_audio

# def transcribe_audio(audio_path):
#     """Transcribes audio using Whisper AI."""
#     result = model.transcribe(audio_path, language="en", fp16=False)
#     return result["segments"]

# def analyze_lip_sync(video_path):
#     """Analyzes lip movements to validate spoken words."""
#     cap = cv2.VideoCapture(video_path)
#     face_data = []
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         try:
#             analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#             face_data.append(analysis[0]["dominant_emotion"])
#         except:
#             face_data.append("neutral")  # Default if no face detected

#     cap.release()
#     return face_data

# def generate_captions(video_path, output_srt="captions.srt"):
#     """Combines Whisper AI transcription with lip sync analysis to generate captions."""
#     audio_file = extract_audio(video_path)
#     segments = transcribe_audio(audio_file)
#     lip_sync_data = analyze_lip_sync(video_path)

#     with open(output_srt, "w") as f:
#         for i, segment in enumerate(segments):
#             start = segment["start"]
#             end = segment["end"]
#             text = segment["text"]

#             # Check lip sync data for better accuracy
#             if "talking" in lip_sync_data[int(start):int(end)]:
#                 confidence = "✅"  # Lip movement detected
#             else:
#                 confidence = "⚠️"  # No clear lip movement

#             f.write(f"{i+1}\n{start:.2f} --> {end:.2f}\n{text} {confidence}\n\n")
    
#     print(f"✅ Captions saved as {output_srt}")

# # Run the script
# video_file = "video.mp4"  # Change to your video file
# generate_captions(video_file)

#<<<<<----------------Working Code------------------>>>>>
# import whisper
# import subprocess  # For using ffmpeg to extract audio

# # Load Whisper model (use "large-v3" for best accuracy)
# model = whisper.load_model("large-v3")

# def extract_audio(video_path, output_audio="audio2.wav"):
#     """Extracts audio from video using ffmpeg."""
#     command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio} -y"
#     subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     return output_audio

# def transcribe_audio(audio_path):
#     """Transcribes audio using Whisper AI."""
#     result = model.transcribe(audio_path, language="en", fp16=False)
#     return result["segments"]

# def generate_captions(video_path, output_srt="captions2.srt"):
#     """Generates captions from audio and saves as an SRT file."""
#     audio_file = extract_audio(video_path)
#     segments = transcribe_audio(audio_file)

#     with open(output_srt, "w") as f:
#         for i, segment in enumerate(segments):
#             start = segment["start"]
#             end = segment["end"]
#             text = segment["text"]

#             f.write(f"{i+1}\n{start:.2f} --> {end:.2f}\n{text}\n\n")
    
#     print(f"✅ Captions saved as {output_srt}")

# # Run the script
# video_file = "video2.mp4"  # Change to your video file
# generate_captions(video_file)
import whisper
import subprocess  # For using ffmpeg to extract audio

# Load Whisper model (use "large-v3" for best accuracy)
model = whisper.load_model("large-v3")

def extract_audio(video_path, output_audio="audio4.wav"):
    """Extracts audio from video using ffmpeg."""
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio} -y"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio

def transcribe_audio(audio_path):
    """Transcribes audio using Whisper AI."""
    result = model.transcribe(audio_path, language="en", fp16=False)
    return result["segments"]

def generate_paragraph_captions(video_path, output_txt="captions3.txt"):
    """Generates captions in a single paragraph and saves as a TXT file."""
    audio_file = extract_audio(video_path)
    segments = transcribe_audio(audio_file)

    full_text = " ".join(segment["text"] for segment in segments)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ Captions saved as {output_txt}")

# Run the script
video_file = "video2.mp4"  # Change to your video file
generate_paragraph_captions(video_file)
