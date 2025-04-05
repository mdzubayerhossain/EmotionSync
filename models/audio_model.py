import torch
from torch import nn

class AudioEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(13, 64, kernel_size=3, padding=1)  # 13 MFCC features
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 431, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 7)  # 7 emotions
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

def save_model(model, path='models/audio_emotion.pth'):
    torch.save(model.state_dict(), path)
