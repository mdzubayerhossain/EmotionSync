# emotionsync.py
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import cv2
from fer import FER
import speech_recognition as sr
import librosa
import torch
from torch import nn
import asyncio
from typing import Dict, List, Tuple
from models.audio_model import AudioEmotionCNN

class EmotionAnalyzer:
    def __init__(self):
        self.face_detector = FER(mtcnn=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.recognizer = sr.Recognizer()
        self.audio_model = AudioEmotionCNN()
        # Load pre-trained weights if available
        try:
            self.audio_model.load_state_dict(torch.load('models/audio_emotion.pth'))
        except FileNotFoundError:
            print("Audio model weights not found. Using untrained model.")

class ContextManager:
    def __init__(self):
        self.conversation_history = []
        self.emotion_history = []
        self.max_history = 10

    def update_context(self, text: str, emotion: Dict):
        self.conversation_history.append(text)
        self.emotion_history.append(emotion)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
            self.emotion_history.pop(0)

class ResponseGenerator:
    def __init__(self):
        self.emotion_responses = {
            'happy': ["Great to see you're feeling good!", "Keeping the positive vibes going!"],
            'sad': ["I'm here for you.", "Want to talk about what's on your mind?"],
            'angry': ["Let's take a deep breath together.", "How can I assist you right now?"]
        }

    def generate_response(self, emotions: Dict, context: List) -> str:
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        return np.random.choice(self.emotion_responses.get(dominant_emotion, ["How can I help you today?"]))

class EmotionSync:
    def __init__(self):
        self.analyzer = EmotionAnalyzer()
        self.context = ContextManager()
        self.response_generator = ResponseGenerator()

    async def process_video_feed(self, frame) -> Dict:
        emotions = self.analyzer.face_detector.detect_emotions(frame)
        return emotions[0]['emotions'] if emotions else {}

    async def process_audio(self, audio_file: str) -> Dict:
        y, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        audio_tensor = torch.FloatTensor(mfcc).unsqueeze(0)
        with torch.no_grad():
            emotion_pred = self.analyzer.audio_model(audio_tensor)
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        scores = torch.softmax(emotion_pred, dim=1)[0].tolist()
        return dict(zip(emotions, scores))

    async def process_text(self, text: str) -> Dict:
        inputs = self.analyzer.tokenizer(text, return_tensors="tf", padding=True, truncation=True)
        outputs = self.analyzer.bert_model(inputs)
        # Simplified emotion classification (placeholder)
        return {"happy": 0.7, "sad": 0.2, "angry": 0.1}

async def main():
    assistant = EmotionSync()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        video_task = asyncio.create_task(assistant.process_video_feed(frame))
        emotions = await video_task
        assistant.context.update_context("User interaction", emotions)
        
        response = assistant.response_generator.generate_response(
            emotions, assistant.context.conversation_history
        )
        print(f"Assistant: {response}")
        
        cv2.imshow('EmotionSync', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
