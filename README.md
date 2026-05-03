# 🧠 Image Caption Generator (PyTorch + Attention)

## 🚀 Overview

This project is an **AI-powered Image Caption Generator** built using **PyTorch**, combining **Computer Vision (CNN)** and **Natural Language Processing (LSTM + Attention)**.

It takes an image as input and generates a meaningful natural language caption describing the scene.

---

## 🔥 Features

- 🖼️ Image Caption Generation using Deep Learning
- 🧠 CNN Encoder (**ResNet50**) for feature extraction
- 🔤 LSTM Decoder for sequence generation
- 🎯 **Attention Mechanism** for better visual understanding
- 📊 Training + Validation pipeline
- ⚡ Top-K Sampling & Greedy Decoding
- 📦 Supports both **Flickr Dataset** and **MS COCO Dataset**
- 💾 Model checkpoint saving

---

## 🏗️ Project Structure

```
image-caption-generator/
│── data/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│
│── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   └── model.py
│
│── utils/
│   ├── vocab.py
│   ├── dataloader.py
│   └── coco_dataset.py
│
│── train.py
│── inference.py
│── requirements.txt
│── README.md
```

---

## 🧠 Model Architecture

### 🔹 Encoder (CNN)

- Pretrained **ResNet50**
- Extracts spatial feature maps from images
- Fine-tuned last layers for better performance

### 🔹 Attention Mechanism

- Allows model to focus on relevant regions of the image
- Improves caption accuracy significantly

### 🔹 Decoder (LSTM)

- Generates captions word-by-word
- Uses context vectors from attention

---

## 📊 Dataset

### ✅ Supported Datasets:

- Flickr8k / Flickr30k
- MS COCO (train2017 + val2017)

### 📂 COCO Structure:

```
data/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

---

## ⚙️ Installation

```bash
git clone <your-repo-link>
cd image-caption-generator

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py
```

### 🔧 Training Details:

- Optimizer: Adam
- Loss: CrossEntropyLoss
- Encoder Fine-tuning (layer4)
- Validation after each epoch

---

## 🧪 Inference

```bash
python inference.py
```

### Example Output:

```
Generated Caption: a man with a backpack standing on a mountain
```

---

## 🎯 Decoding Strategies

- Greedy Decoding (deterministic)
- Top-K Sampling (more diverse captions)

---

## 📈 Results

| Model Version | Description                         |
| ------------- | ----------------------------------- |
| Baseline      | CNN + LSTM                          |
| Improved      | + Attention                         |
| Final         | + Fine-tuned Encoder + COCO Dataset |

---

## ⚠️ Challenges & Learnings

- Dataset bias (common objects like dogs appear frequently)
- Importance of attention in vision-language tasks
- Trade-off between randomness and accuracy in decoding
- Handling large-scale datasets like COCO

---

## 🚀 Future Improvements

- 🔥 Beam Search Decoding
- 🌐 Deploy using Streamlit / Web App
- 🤖 Compare with pretrained models (Hugging Face)
- 📱 Android App integration (planned)
- ⚡ Faster training with GPU

---

## 🧑‍💻 Tech Stack

- Python
- PyTorch
- Torchvision
- NLTK
- PIL

---

## 📌 Author

**Tanish Gupta**
AI | ML | IoT | Robotics Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

---
