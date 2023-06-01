import ast
import json

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from utils import get_embedding  # type: ignore


@torch.no_grad()
def find_similar_knn(cur_embedding, weights, k=10, least = False):
    # sourcery skip: boolean-if-exp-identity, remove-unnecessary-cast
    # cur_embedding = get_embedding(paper)
    assert(torch.is_tensor(cur_embedding))
    # print(cur_embedding.shape)
    assert(cur_embedding.shape==(1,768))
    # weights_norm = F.normalize(weights, p=2, dim=1).double() # (N, d)
    # cur_em_norm = F.normalize(cur_embedding, p=2, dim=1).double() # (1, d)
    cos_sim = F.cosine_similarity(weights, cur_embedding, dim=1)
    topk = torch.topk(cos_sim, k, largest = False if least else True)
    top_indices = topk.indices
    top_values = topk.values
    return top_indices, top_values

def evaluate(recommended, actual, k=None):
    recommended = np.asarray(recommended)[:k] if k else np.asarray(recommended)
    actual = np.asarray(actual)

    true_positives = np.intersect1d(recommended, actual)
    false_positives = np.setdiff1d(recommended, actual)
    false_negatives = np.setdiff1d(actual, recommended)

    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def evaluate_knn(pid_list, embedding_map, reference_map, k=20):
    precisions = []
    recalls = []
    # temp_cnt = 0
    for paper_id in pid_list:
        paper_embedding = embedding_map[paper_id]
        top_indices, top_values = find_similar_knn(paper_embedding.view(1,-1), weights, k=k, least=False)
        recommendations = {all_paper_ids[i] for i in top_indices}
        # recommendations = set(get_recommendations(model, paper_id, embedding_map, k=k))
        true_references = set(reference_map.get(paper_id, []))
        if not true_references:
            continue

        # intersect = recommendations.intersection(true_references)
        # precision = len(intersect) / len(recommendations)
        # recall = len(intersect) / len(true_references)

        precision, recall, f1_score = evaluate(list(recommendations), list(true_references), k=k)
        # print(f'({len(recommendations)},{len(true_references)}){precision=}    {recall=}')

        precisions.append(precision)
        recalls.append(recall)
        # temp_cnt += 1
        # if temp_cnt == 10:
        #     break

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    return mean_precision, mean_recall


if __name__ == '__main__':


    print(f'Cuda: {torch.cuda.is_available()}')
    print(f'PyTorch: {torch.__version__}')
    print(f'Device count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'Device name: {torch.cuda.get_device_name(0)}')

    DATA_PATH = 'data/specter'

    with open(f'{DATA_PATH}/output_10k.json', 'r') as f:
        embeddings = [json.loads(line) for line in f]

    with open(f'{DATA_PATH}/metadata_10k_full.json', 'r') as f:
        metadata = json.load(f)

    weights = torch.stack([torch.tensor(e['embedding']) for e in embeddings]).double()
    all_paper_ids = [e['paper_id'] for e in embeddings]

    specter_tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    specter_model = AutoModel.from_pretrained('allenai/specter')

    K = 20

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

    test_paper_ids = []
    with open(f"{DATA_PATH}/test.txt", "r") as f:
        for line in f:
            id = line.strip()
            test_paper_ids.append(id)

    # precision, recall = evaluate_knn(test_paper_ids, embedding_map, reference_map, k=K)
    # print(f"Precision @ P{K}: {precision}")
    # print(f"Recall @ {K}: {recall}")

    paper_id = '204e3073870fae3d05bcbc2f6a8e263d9b72e776'
    paper_embedding = get_embedding(metadata[paper_id], specter_tokenizer=specter_tokenizer, specter_model=specter_model)
    top_indices, top_values = find_similar_knn(paper_embedding, weights, k=50, least=False)
    recommended_paper_ids = [all_paper_ids[i] for i in top_indices]


    cnt = 0
    for paper_id, cos_sim in zip(recommended_paper_ids, top_values):
        title = metadata[paper_id]['title']
        # abstract = metadata[paper_id]['abstract']
        year = metadata[paper_id]['year']
        print(f'Paper ID: {paper_id}\nTitle: {title}\nYear: {year}\nCosine similarity: {cos_sim}\n')
        cnt += 1
        if cnt == 10:
            break

    actual_references = ast.literal_eval(metadata['204e3073870fae3d05bcbc2f6a8e263d9b72e776'].get('references'))

    print('-'*100)

    precision, recall, f1_score = evaluate(recommended_paper_ids[1:], actual_references, k=K)
    print(f"Precision @ {K}: {precision}")
    print(f"Recall @ {K}: {recall}")
    print(f"F1 Score: {f1_score}")