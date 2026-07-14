import open_clip

print("openclip imported")

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device="cpu"
)

print("model loaded")

import faiss

print("faiss imported")