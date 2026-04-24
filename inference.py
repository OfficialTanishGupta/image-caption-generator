import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

from models.model import ImageCaptionModel
from utils.vocab import Vocabulary

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Load Vocabulary
# ======================
vocab = Vocabulary()
df = pd.read_csv("data/captions.txt")
vocab.build_vocab(df["caption"].tolist())
vocab_size = len(vocab)

# ======================
# Load Model
# ======================
model = ImageCaptionModel(256, 256, vocab_size).to(device)
model.load_state_dict(torch.load("model_epoch_10.pth", map_location=device))
model.eval()

# ======================
# Image Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ======================
# Top-K Sampling (optional)
# ======================
def sample_top_k(logits, k=5):
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k)

    top_k_probs = top_k_probs.squeeze()
    top_k_indices = top_k_indices.squeeze()

    sampled_idx = torch.multinomial(top_k_probs, 1)
    predicted = top_k_indices[sampled_idx]

    return predicted.item()

# ======================
# Caption Generator
# ======================
def generate_caption(image_path, max_length=20, k=None):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    result_caption = []

    with torch.no_grad():
        features = model.encoder(image)

        # Initialize hidden states
        h = torch.zeros(1, 256).to(device)
        c = torch.zeros(1, 256).to(device)

        # Start token
        inputs = torch.tensor([vocab.stoi["<sos>"]]).to(device)

        for _ in range(max_length):
            embeddings = model.decoder.embedding(inputs).unsqueeze(0)

            # Attention
            context, _ = model.decoder.attention(features, h)

            # LSTM input
            lstm_input = torch.cat((embeddings.squeeze(0), context), dim=1)

            h, c = model.decoder.lstm(lstm_input, (h, c))

            output = model.decoder.fc(h)

            # Sampling strategy
            if k is not None:
                predicted = sample_top_k(output, k)
            else:
                predicted = torch.argmax(output, dim=1).item()

            word = vocab.itos.get(predicted, "<unk>")

            if word == "<eos>":
                break

            result_caption.append(word)

            inputs = torch.tensor([predicted]).to(device)

    return " ".join(result_caption)

# ======================
# Run Inference
# ======================
if __name__ == "__main__":
    test_image_path = "Test_Images/Image.jpg"

    try:
        # You can try k=5 or remove it
        caption = generate_caption(test_image_path, k=5)
        print("\nGenerated Caption:", caption)

    except FileNotFoundError:
        print(f"\nError: Could not find image at {test_image_path}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")