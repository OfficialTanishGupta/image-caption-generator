import os
from dotenv import load_dotenv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Load env and token
load_dotenv()
my_token = os.getenv("HF_TOKEN")

# 2. Load model and processor directly
print("Loading model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", token=my_token)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", token=my_token)

# 3. Load your local image
# Change 'test_image.jpg' to the actual name of the file you uploaded
img_path = r"C:\Users\Tanish_Gupta\OneDrive\Desktop\ML Projects\image-caption-generator\Test_Images\hiker-GRTE-NPS.jpg"
try:
    raw_image = Image.open(img_path).convert('RGB')
    print(f"Successfully loaded {img_path} from your folder!")
except FileNotFoundError:
    print(f"Error: The file '{img_path}' was not found in this folder.")
    exit()

# 4. Process and generate
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)

# 5. Result
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"\nResult: {caption}")
