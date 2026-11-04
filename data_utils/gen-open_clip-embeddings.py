from pathlib import Path

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

# 1. Load the OpenCLIP RN50 model
# Common pre-trained weights include 'openai' or 'yfcc15m'
model_name = "RN50"
pretrained_weights = "openai"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained_weights
)
model.eval()

# 2. Define stimuli directory and gather all image file paths
stimuli_dir = (
    "/Users/emdupre/Desktop/UdeM-Projects/things-encode/cneuromod.things.stimuli"
)
stimuli_fpaths = list(Path(stimuli_dir).rglob("*.jpg"))

# 3. Prepare a container to store the pooled layers
batch_layer_outputs = {}
all_filenames = []
all_embeddings = {"layer1": [], "layer2": [], "layer3": [], "layer4": [], "final": []}


# 4. Register hooks on the 4 main ResNet stages
# open_clip's RN50 visual backbone resides in model.visual
def get_pooling_hook(layer_name):
    def hook(module, input, output):
        # Global Average Pooling (GAP) sur les dimensions spatiales [B, C, H, W] -> [B, C]
        pooled_output = output.mean(dim=[2, 3])
        batch_layer_outputs[layer_name] = pooled_output.detach().cpu()

    return hook


hooks = [
    model.visual.layer1.register_forward_hook(get_pooling_hook("layer1")),  # 256
    model.visual.layer2.register_forward_hook(get_pooling_hook("layer2")),  # 512
    model.visual.layer3.register_forward_hook(get_pooling_hook("layer3")),  # 1024
    model.visual.layer4.register_forward_hook(get_pooling_hook("layer4")),  # 2048
]

# 6. Boucle d'extraction principale
with torch.no_grad():
    for img_path in tqdm(stimuli_fpaths, desc="Processing images"):
        # Forward pass (les hooks vont intercepter layer1 à layer4)
        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)
        final_emb = model.encode_image(image_tensor)

        # Stockage du batch actuel
        all_filenames.append(img_path.name)
        all_embeddings["final"].append(final_emb)  # 1024

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            all_embeddings[layer_name].append(batch_layer_outputs[layer_name])

# Retirer les hooks proprement
for hook in hooks:
    hook.remove()

# 7. Concaténation de tous les lots en un seul grand tenseur par couche
for layer_name in all_embeddings.keys():
    all_embeddings[layer_name] = torch.cat(all_embeddings[layer_name], dim=0)

# 8. Sauvegarde des résultats sur le disque
# Vous obtenez un dictionnaire contenant les tenseurs finaux ainsi que l'ordre des fichiers
output_data = {"filenames": all_filenames, "embeddings": all_embeddings}
torch.save(output_data, "clip_rn50_all_layers.pt")

# this can be later loaded with:
# loaded_data = torch.load("clip_rn50_all_layers.pt")
# and accessed as:
# loaded_data["filenames"]
