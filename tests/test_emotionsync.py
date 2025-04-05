import pytest
import numpy as np
from emotionsync import EmotionAnalyzer, ContextManager, ResponseGenerator, EmotionSync
import cv2
import torch

@pytest.fixture
def emotionsync():
    return EmotionSync()

def test_emotion_analyzer_init():
    analyzer = EmotionAnalyzer()
    assert analyzer.face_detector is not None
    assert analyzer.tokenizer is not None
    assert analyzer.bert_model is not None
    assert analyzer.audio_model is not None

def test_context_manager(emotionsync):
    emotionsync.context.update_context("Hello", {"happy": 0.8})
    assert len(emotionsync.context.conversation_history) == 1
    assert emotionsync.context.emotion_history[0]["happy"] == 0.8

def test_response_generator():
    generator = ResponseGenerator()
    response = generator.generate_response({"happy": 0.8, "sad": 0.1}, [])
    assert response in generator.emotion_responses['happy']

@pytest.mark.asyncio
async def test_video_processing(emotionsync):
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    emotions = await emotionsync.process_video_feed(frame)
    assert isinstance(emotions, dict)

@pytest.mark.asyncio
async def test_audio_processing(emotionsync):
    # Use a small dummy audio file or generate one
    audio_data = np.random.randn(16000)  # 1 second at 16kHz
    librosa.output.write_wav('test.wav', audio_data, 16000)
    emotions = await emotionsync.process_audio('test.wav')
    assert len(emotions) == 7  # 7 emotion categories
    assert sum(emotions.values()) == pytest.approx(1.0, 0.1)  # Softmax sum

def test_full_integration(emotionsync):
    emotions = {"happy": 0.8, "sad": 0.1, "angry": 0.05}
    emotionsync.context.update_context("Test", emotions)
    response = emotionsync.response_generator.generate_response(emotions, emotionsync.context.conversation_history)
    assert isinstance(response, str)
