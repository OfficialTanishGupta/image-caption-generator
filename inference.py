import torch
from PIL import Image
import torchvision.transforms as transforms

from models.model import ImageCaptionModel
from utils.vocab import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab (rebuild same way)
vocab = Vocabulary()
# You MUST rebuild same vocab from captions
import pandas as pd
df = pd.read_csv("data/captions.txt")
vocab.build_vocab(df["caption"].tolist())

vocab_size = len(vocab)

# Load model
model = ImageCaptionModel(256, 256, vocab_size).to(device)
model.load_state_dict(torch.load("model_epoch_2.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def generate_caption(image_path, max_length=20):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    result_caption = []

    with torch.no_grad():
        features = model.encoder(image)

        # Start with <sos>
        inputs = torch.tensor([[vocab.stoi["<sos>"]]]).to(device)

        for _ in range(max_length):
            outputs = model.decoder(features, inputs)

            predicted = outputs[:, -1, :].argmax(1).item()
            word = vocab.itos.get(predicted, "<unk>")

            if word == "<eos>":
                break

            result_caption.append(word)

            # 🔥 KEY STEP: feed prediction back
            inputs = torch.cat(
                [inputs, torch.tensor([[predicted]]).to(device)],
                dim=1
            )

    return " ".join(result_caption)

# Test
print(generate_caption("Test_Images/hiker-GRTE-NPS.jpg"))