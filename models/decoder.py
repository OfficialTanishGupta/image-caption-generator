import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)

        features = features.unsqueeze(0)
        
        embeddings = embeddings.permute(1, 0, 2)

        inputs = torch.cat((features, embeddings), dim=0)

        outputs, _ = self.lstm(inputs)
        
        outputs = self.fc(outputs)

        return outputs.permute(1, 0, 2)
