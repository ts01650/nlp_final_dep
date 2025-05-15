import torch
import json
import numpy as np
from model_architecture import BiGRUCRFClass

def load_model(model_path):
    embedding_matrix = np.load("embedding_matrix.npy")
    hidden_dim = 256
    output_dim = 4
    model = BiGRUCRFClass(embedding_matrix, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict(model, sentence, label_map):
    words = sentence.strip().split()
    idx_map = {w: i+2 for i, w in enumerate(label_map)}  # dummy word2idx
    input_ids = [idx_map.get(w.lower(), 1) for w in words]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    mask = torch.tensor([[1]*len(input_ids)], dtype=torch.uint8)
    with torch.no_grad():
        outputs = model(input_tensor, mask=mask)
    label_inv = {v: k for k, v in label_map.items()}
    tags = [label_inv.get(tag, "O") for tag in outputs[0]]
    return list(zip(words, tags))
