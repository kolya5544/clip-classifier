import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Load CLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Load ImageNet-1k labels
f = open("imagenet_classes.txt", "r")
imagenet_classes = f.readlines()
imagenet_classes = list(map(lambda s: s.strip(), imagenet_classes))

# Tokenize ImageNet labels
text_inputs = tokenizer(imagenet_classes)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)


@app.post("/upload/")
async def create_upload_file(file: UploadFile):
    image = preprocess(Image.open(file.file)).unsqueeze(0)

    # Compute similarity between image and labels
    with torch.no_grad():
        image_features = model.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)  # Cosine similarity

    # Get top 5 predictions
    top_indices = similarity.argsort(descending=True)[:5]
    top_labels = [imagenet_classes[i] for i in top_indices]

    return top_labels


