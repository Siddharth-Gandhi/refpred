import ast
import json
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import PaperPairDataset  # type: ignore
from knn import find_similar_knn, get_embedding  # type: ignore
from model import PaperPairModel  # type: ignore
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def get_recommendations(model, paper_id, embedding_map, k=20):
    # model.eval()
    paper_embedding = embedding_map[paper_id]
    # other_paper_ids = [pid for pid in embedding_map if pid != paper_id]
    # paper_embedding = get_embedding(metadata[paper_id])
    top_indices, top_values = find_similar_knn(paper_embedding.view(1,-1), weights, k=50, least=False)
    recommended_paper_ids = [all_paper_ids[i] for i in top_indices]
    other_paper_embeddings = torch.stack([embedding_map[pid] for pid in recommended_paper_ids])

    with torch.no_grad():
        paper_embedding = paper_embedding.expand_as(other_paper_embeddings).to(DEVICE)
        other_paper_embeddings = other_paper_embeddings.to(DEVICE)
        scores = model(paper_embedding, other_paper_embeddings)
    # model.train()
    top_k_indices = torch.topk(scores, k=k).indices
    return [recommended_paper_ids[idx] for idx in top_k_indices]


def evaluate_model(model, embedding_map, reference_map, k=20):
    precisions = []
    recalls = []
    temp_cnt = 0
    for paper_id in embedding_map:

        recommendations = set(get_recommendations(model, paper_id, embedding_map, k=k))
        true_references = set(reference_map.get(paper_id, []))
        if not true_references:
            continue

        intersect = recommendations.intersection(true_references)
        precision = len(intersect) / len(recommendations)
        recall = len(intersect) / len(true_references)

        precisions.append(precision)
        recalls.append(recall)
        temp_cnt += 1
        if temp_cnt == 10:
            break

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    return mean_precision, mean_recall


if __name__ == '__main__':


    print(f'Cuda: {torch.cuda.is_available()}')
    print(f'PyTorch: {torch.__version__}')
    print(f'Device count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'Device name: {torch.cuda.get_device_name(0)}')

    DATA_PATH = 'data/specter/'

    with open(f'{DATA_PATH}/output_10k.json', 'r') as f:
        embeddings = [json.loads(line) for line in f]

    with open(f'{DATA_PATH}/metadata_10k_full.json', 'r') as f:
        metadata = json.load(f)

    weights = torch.stack([torch.tensor(e['embedding']) for e in embeddings]).double()
    all_paper_ids = [e['paper_id'] for e in embeddings]

    specter_tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    specter_model = AutoModel.from_pretrained('allenai/specter')

    # Hyperparameters
    K = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 5

    embedding_map = {}
    for obj in embeddings:
        # object is a dict like {'paper_id': str, 'embedding': np.array}
        paper_id, emb_768 = obj.values()
        arr = np.asarray(emb_768)
        embedding_map[paper_id] = torch.tensor(arr, dtype=torch.float32)

    reference_map = {}
    # fetch references from metadata dict in the format {paper_id: [list of references]}
    for paper_id in all_paper_ids:
        references = ast.literal_eval(metadata[paper_id].get('references'))
        reference_map[paper_id] = references or []

    # Instantiate the dataset, model, loss function, and optimizer
    dataset = PaperPairDataset(embedding_map, reference_map)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = PaperPairModel()
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Train the model


    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            embeddings1, embeddings2, labels = data
            embeddings1 = embeddings1.to(DEVICE)
            embeddings2 = embeddings2.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(embeddings1, embeddings2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    precision, recall = evaluate_model(model, embedding_map, reference_map, k=20)
    print(f"Precision @ 20: {precision}")
    print(f"Recall @ 20: {recall}")

    # test_paper_ids = []
    # with open(f"{DATA_PATH}/test.txt", "r") as f:
    #     for line in f:
    #         id = line.strip()
    #         test_paper_ids.append(id)



    # #TODO precision, recall = evaluate_knn(test_paper_ids, embedding_map, reference_map, k=K)
    # print(f"Precision @ P{K}: {precision}")
    # print(f"Recall @ {K}: {recall}")

    # paper_id = '204e3073870fae3d05bcbc2f6a8e263d9b72e776'
    # paper_embedding = get_embedding(metadata[paper_id])
    # top_indices, top_values = find_similar_knn(paper_embedding, weights, k=50, least=False)
    # recommended_paper_ids = [all_paper_ids[i] for i in top_indices]


    # cnt = 0
    # for paper_id, cos_sim in zip(recommended_paper_ids, top_values):
    #     title = metadata[paper_id]['title']
    #     # abstract = metadata[paper_id]['abstract']
    #     year = metadata[paper_id]['year']
    #     print(f'Paper ID: {paper_id}\nTitle: {title}\nYear: {year}\nCosine similarity: {cos_sim}\n')
    #     cnt += 1
    #     if cnt == 10:
    #         break

    # actual_references = ast.literal_eval(metadata['204e3073870fae3d05bcbc2f6a8e263d9b72e776'].get('references'))

    # print('-'*100)

    # precision, recall, f1_score = evaluate(recommended_paper_ids[1:], actual_references, k=K)
    # print(f"Precision @ {K}: {precision}")
    # print(f"Recall @ {K}: {recall}")
    # print(f"F1 Score: {f1_score}")