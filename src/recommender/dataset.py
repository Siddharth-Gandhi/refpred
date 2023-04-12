import random

import torch
from torch.utils.data import Dataset


# TODO: Prepare from KNN results
class PaperPairDataset(Dataset):
    def __init__(self, embedding_map, reference_map, num_positives=5, num_negatives=5):
        self.embedding_map = embedding_map
        self.reference_map = reference_map
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.paper_ids = list(embedding_map.keys())
        self.prepare_random_data()

    def prepare_random_data(self):
        self.data = []
        for paper_id in self.paper_ids:
            if paper_id in self.reference_map:
                pos_count = 0
                for ref_id in self.reference_map[paper_id]:
                    if ref_id in self.embedding_map:
                        self.data.append((paper_id, ref_id, 1))
                        pos_count += 1
                        if pos_count >= self.num_positives:
                            break

            neg_count = 0
            while neg_count < self.num_negatives:
                random_id = random.choice(self.paper_ids)
                if random_id not in self.reference_map.get(paper_id, []) and random_id != paper_id:
                    self.data.append((paper_id, random_id, 0))
                    neg_count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paper_id1, paper_id2, score = self.data[idx]
        embedding1 = self.embedding_map[paper_id1]
        embedding2 = self.embedding_map[paper_id2]
        return torch.tensor(embedding1), torch.tensor(embedding2), torch.tensor(score, dtype=torch.float32)
