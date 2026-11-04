from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Load DINOv2 model and processor
# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# model = AutoModel.from_pretrained("facebook/dinov2-base")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
model = AutoModel.from_pretrained("facebook/dinov2-giant")
model.eval()


def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling over the last hidden state patches
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding


# Example usage:
# vector = get_embedding("path_to_image.jpg")

embeddings = []
file_names = []
stimuli_dir = (
    "/Users/emdupre/Desktop/UdeM-Projects/things-encode/cneuromod.things.stimuli"
)

for img_path in tqdm(Path(stimuli_dir).rglob("*.jpg"), desc="Processing images"):
    vector = get_embedding(img_path)

    embeddings.append(vector)
    file_names.append(img_path.name)

# Save embeddings and file names to .npy files
np.save("DINOv2-giant-embeddings.npy", np.array(embeddings))
np.savetxt("DINOv2-giant-file_names.txt", np.array(file_names), fmt="%s")
