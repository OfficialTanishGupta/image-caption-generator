import torch
import torch.nn as nn
from models.attention import Attention

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, feature_dim=2048):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(feature_dim, hidden_size)

        self.lstm = nn.LSTMCell(embed_size + feature_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        batch_size = features.size(0)
        seq_length = captions.size(1)

        embeddings = self.embedding(captions)

        h, c = (
            torch.zeros(batch_size, self.lstm.hidden_size).to(features.device),
            torch.zeros(batch_size, self.lstm.hidden_size).to(features.device),
        )

        outputs = []

        for t in range(seq_length):
            context, _ = self.attention(features, h)

            lstm_input = torch.cat((embeddings[:, t, :], context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))

            output = self.fc(self.dropout(h))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)

        return outputs