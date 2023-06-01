import ast
import json
import math

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.neighbors import BallTree
from transformers import AutoModel, AutoTokenizer

from src.recommender.model import PaperPairModel

# Import your get_recommendations function here
# from your_module import get_recommendations

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"*": {"origins": "*"}})


DATA_PATH = "data/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hparams = {
    "batch_size": 2048,
    # 'hidden_dim': 768,
    # 'num_hidden_layers': 3,
    "hidden_dims": [1024, 2048],
    "lr": 1e-2,
    "min_lr": 1e-7,
    "patience": 3,
    "factor": 0.1,
    "num_epochs": 250,
    "weight_decay": 1e-3,
    "dropout_prob": 0.5,
    "stop_after": 5,  # does NOT work, need to fix
    "knn_k": 1000,  # for getting nearest neighbours
    "top_k": 20,  # for evaluation
    "use_bn": True,
    "bn_momentum": 0.9,
}

with open(f"{DATA_PATH}/output_10k.json", "r") as f:
    embeddings = [json.loads(line) for line in f]

with open(f"{DATA_PATH}/references_output_100k.json", "r") as f:
    reference_embeddings = [json.loads(line) for line in f]

with open(f"{DATA_PATH}/metadata_10k_full.json", "r") as f:
    metadata = json.load(f)

with open(f"{DATA_PATH}/references_metadata_formatted.json", "r") as f:
    reference_metadata = json.load(f)

# remove ref_ids with null year
all_ref_ids = [e["paper_id"] for e in reference_embeddings]

# removing some papers which are buggy in S2 API
s2_bug_pids = {
    "3e83d54c5e8dfba82638b4f75ace31505ea60ff0",
    "9dd051e6f842131196fee5cbc79b8e4511d577c2",
    "817aa71dd75abc01dedb24f806d69e8e97828a11",
    "16c232a9310860be9e9817cca875cd72d9ba50d4",
    "468c3b2bf358d07cc625b075f91595d825299948",
    "022dd244f2e25525eb37e9dda51abb9cd8ca8c30",
    "0d684d919652ab2506fc8ef0a2494a46c3f7abca",
    "21b770571687a483672894374065b93e246fd200",
    "b281a8a5f9af12143b0813ebe65eac3e9971f316",
    "bd33916225d23a8855a1e67ae73321d7b70fcd0c",
    "7cccee8c8a3807b1699b1b82bdaa8e5e66eb5d0f",
    "bc1586a2e74d6d1cf87b083c4cbd1eede2b09ea5",
    "6e0cfc8a2e743e3a90ad089f0fd4e4985f2f6834",
    "0aea520a25198f6b3f385a09b158da2f7ec5cf1f",
    "7c53d9c66a8648abb060318e36be4266233c4c0c",
    "6e45220c1f3a8a8cbf176a2fc722c7e8380d5dd4",
    "98485ce6532d69f34a8ec67de6b09a39532bd221",
    "dfc504536e8434eb008680343abb77010965169e",
}
all_ref_ids = list(set(all_ref_ids) - s2_bug_pids)

null_year_ref_ids = [pid for pid in all_ref_ids if reference_metadata[pid]["year"] is None]
null_year_ref_ids_set = set(null_year_ref_ids)
all_ref_ids = list(set(all_ref_ids) - null_year_ref_ids_set)

reference_metadata = {k: v for k, v in reference_metadata.items() if k not in null_year_ref_ids_set}
reference_embeddings = [
    e for e in reference_embeddings if e["paper_id"] not in null_year_ref_ids_set
]

all_paper_ids = [e["paper_id"] for e in embeddings]
# zero_ref_pids = [
#     pid for pid in all_paper_ids if len(ast.literal_eval(metadata[pid]["references"])) == 0
# ]
zero_ref_pids = [pid for pid in all_paper_ids if len(metadata[pid]["references"]) == 0]
all_paper_ids = list(set(all_paper_ids) - set(zero_ref_pids))
all_pid_set = set(all_paper_ids)

common_pid_set = set(all_paper_ids) & set(all_ref_ids)

all_pids = all_paper_ids + [pid for pid in all_ref_ids if pid not in common_pid_set]


embeddings = [e for e in embeddings if e["paper_id"] in all_pid_set]
metadata = {k: v for k, v in metadata.items() if k in all_pid_set}


for p in metadata.values():
    # p["references"] = ast.literal_eval(p["references"])
    p["references"] = p["references"]

metadata = {**metadata, **{k: v for k, v in reference_metadata.items() if k not in metadata}}

embedding_map = {}
for obj in embeddings:
    # object is a dict like {'paper_id': str, 'embedding': np.array}
    paper_id, emb_768 = obj.values()
    arr = np.asarray(emb_768)
    embedding_map[paper_id] = torch.tensor(arr, dtype=torch.float32)

for obj in reference_embeddings:
    # object is a dict like {'paper_id': str, 'embedding': np.array}
    paper_id, emb_768 = obj.values()
    arr = np.asarray(emb_768)
    embedding_map[paper_id] = torch.tensor(arr, dtype=torch.float32)

embedding_matrix = np.array([embedding_map[paper_id].numpy() for paper_id in all_pids])

knn_tree = BallTree(embedding_matrix, metric="l2")

reference_map = {}
# fetch references from metadata dict in the format {paper_id: [list of references]}
for paper_id in all_paper_ids:
    # references = ast.literal_eval(metadata[paper_id].get('references'))
    references = metadata[paper_id].get("references")
    reference_map[paper_id] = references or []

specter_tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
specter_model = AutoModel.from_pretrained("allenai/specter")

############################################################################################################################


@torch.no_grad()
def get_embedding(paper, embedding_map=None):
    """
    Given an input paper (dict with at least 'title' as a key, returns a 768 dimensional embeddings using the SPECTER model from HF.
    """
    assert (
        type(paper) == dict and "title" in paper.keys()
    ), "paper must be a dict with at least 'title' as a key"
    if (
        embedding_map
        and "paper_id" in paper
        and paper["paper_id"] is not None
        and paper["paper_id"] in embedding_map
    ):
        return embedding_map[paper["paper_id"]].view(1, -1)
    # tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    # model = AutoModel.from_pretrained('allenai/specter')
    title_abs = [
        d["title"] + specter_tokenizer.sep_token + (d.get("abstract") or "") for d in [paper]
    ]
    inputs = specter_tokenizer(
        title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    result = specter_model(**inputs)
    cur_embedding = result.last_hidden_state[:, 0, :]
    return cur_embedding


def find_knn(cur_embedding, knn_tree, knn_k, least=False):
    # current_embeddings shape: [1,768]
    # assert(torch.is_tensor(cur_embedding))
    # print(cur_embedding.shape)
    # assert(cur_embedding.shape==(1,768))

    # this score isn't really a score but distance rather (not cosine similarity)
    # this becaues BallTree don't support cosine distance :(
    scores, top_indices = knn_tree.query(
        cur_embedding.reshape(1, -1), k=knn_k, return_distance=True
    )
    scores, top_indices = np.squeeze(scores), np.squeeze(top_indices)
    return top_indices, scores


def get_recommendations(
    model, paper_obj, knn_tree, embedding_map, metadata, all_pids, top_k, knn_k
):
    model.eval()
    # paper_embedding = embedding_map[paper_id]
    paper_embedding = get_embedding(paper_obj, embedding_map)
    top_indices, scores = find_knn(paper_embedding, knn_tree, knn_k=knn_k, least=False)
    recommended_paper_ids = [all_pids[i] for i in top_indices]

    other_paper_embeddings = torch.stack([embedding_map[pid] for pid in recommended_paper_ids])

    # Get years from metadata and create is_after tensor
    paper_year = paper_obj["year"]
    other_paper_years = [metadata[pid]["year"] for pid in recommended_paper_ids]
    is_after = torch.tensor(
        [int(paper_year < other_year) for other_year in other_paper_years], dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        paper_embedding = paper_embedding.expand_as(other_paper_embeddings).to(DEVICE)
        other_paper_embeddings = other_paper_embeddings.to(DEVICE)
        # Pass the is_after tensor to the model
        scores = model(paper_embedding, other_paper_embeddings, is_after)
    model.train()
    top_k = torch.topk(scores, k=top_k)
    # top_k_indices = torch.topk(scores, k=top_k).indices
    top_k_indices = top_k.indices
    top_k_scores = top_k.values
    reranked_pids = [recommended_paper_ids[idx] for idx in top_k_indices]
    return reranked_pids, top_k_scores


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    title = data.get("title")  # type: ignore
    abstract = data.get("abstract")  # type: ignore
    num_papers = int(data.get("num_papers"))  # type: ignore
    # model = PaperPairModel(hidden_dim=hparams['hidden_dim'], num_hidden_layers = hparams['num_hidden_layers'], dropout_prob = hparams['dropout_prob'], use_bn=hparams['use_bn'], bn_momentum=hparams['bn_momentum']) # type: ignore
    model = PaperPairModel(
        hidden_dims=hparams["hidden_dims"],
        dropout_prob=hparams["dropout_prob"],
        use_bn=hparams["use_bn"],
        bn_momentum=hparams["bn_momentum"],
    )
    checkpoint_path = "best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    model.to(DEVICE)

    paper_obj = {"title": title, "abstract": abstract, "year": 2023}
    recommendations_ids, scores = get_recommendations(
        model,
        paper_obj,
        knn_tree,
        embedding_map,
        metadata,
        all_pids,
        top_k=num_papers,
        knn_k=hparams["knn_k"],
    )
    # Call your Python function here and get the recommendations
    # recommendations = get_recommendations(title, abstract)

    # Form the final recommenddations list which contains dict for each paper in the format {'paper_id': str, 'score': float, 'title': str, 'abstract': str, 'year': int, 'authors': str, 'url': str} all of which are fetched from metadata dict
    recommendations = []  # Replace this with the actual recommendations
    for paper_id in recommendations_ids:
        paper = metadata[paper_id]
        # paper['score'] = math.trunc(scores[recommendations_ids.index(paper_id)].item() * 10000) / 10000
        paper["score"] = scores[recommendations_ids.index(paper_id)].item()
        print(paper["score"])
        # if paper['abstract'] is None:
        #     paper['abstract'] = 'Not Available'
        recommendations.append(paper)

    return jsonify(recommendations)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1")
