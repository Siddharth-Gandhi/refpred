# 26th April Latest Working
class PaperPairDataset(Dataset):
    def __init__(self, paper_ids, embedding_map, reference_map, metadata, all_paper_ids, knn_k, knn_tree = None, check_year=True, data_file = None):
        self.paper_ids = paper_ids
        self.embedding_map = embedding_map
        self.reference_map = reference_map
        self.metadata = metadata
        self.all_paper_ids = all_paper_ids
        self.knn_k = knn_k
        self.knn_tree = knn_tree
        self.check_year = check_year

        # if data_file and os.path.exists(data_file):
        #     with open(data_file, 'rb') as f:
        #         self.data = pickle.load(f)
        # else:
        #     self.prepare_data_knn(knn_k=self.knn_k)
        #     if data_file:
        #         with open(data_file, 'wb') as f:
        #             pickle.dump(self.data, f)

        if data_file and os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.data = saved_data['data']
                self.max_refs_present = saved_data['max_refs_present']
                self.total_refs = saved_data['total_refs']
                self.pos_neg_ratio = saved_data['pos_neg_ratio']
        else:
            self.prepare_data_knn(knn_k=self.knn_k)
            if data_file:
                with open(data_file, 'wb') as f:
                    saved_data = {
                        'data': self.data,
                        'max_refs_present': self.max_refs_present,
                        'total_refs': self.total_refs,
                        'pos_neg_ratio': self.pos_neg_ratio
                    }
                    pickle.dump(saved_data, f)


    def prepare_data_knn(self, knn_k):
        self.data = []
        total_pos = 0
        total_neg = 0

        self.max_refs_present = []
        self.total_refs = []
        zero_ref_papers = 0
        for i, paper_id in enumerate(tqdm(self.paper_ids)):
            pos_count = 0
            neg_count = 0
            original_year = self.metadata[paper_id]['year']
            assert type(original_year) == int

            recommendations = find_knn_pid(paper_id, self.knn_tree, self.embedding_map, self.metadata, self.all_paper_ids, knn_k=self.knn_k)
            actual_references = set(self.reference_map[paper_id])

            num_positives = len(actual_references & set(recommendations))
            num_negatives = num_positives

            self.max_refs_present.append(num_positives)
            self.total_refs.append(len(actual_references))

            if not actual_references:
                zero_ref_papers += 1

            for ref_id in recommendations:
                reference_year = self.metadata[ref_id]['year']
                assert type(reference_year) == int

                if pos_count >= num_positives and neg_count >= num_negatives:
                    break

                if ref_id in actual_references and pos_count < num_positives:
                    # a positive example
                    self.data.append((paper_id, ref_id, 1))
                    pos_count += 1
                elif self.check_year and reference_year > original_year and neg_count < num_negatives:
                    # skip KNN receommendations that were not present at the time of the paper
                    self.data.append((paper_id, ref_id, 0))
                    neg_count += 1
                    # print(f'Skipping {ref_id} because it was published after {paper_id} ({reference_year} > {original_year})')
                elif ref_id not in actual_references and neg_count < num_negatives:
                    # a negative example
                    self.data.append((paper_id, ref_id, 0))
                    neg_count += 1
                else:
                    continue

            total_pos += pos_count
            total_neg += neg_count

        self.pos_neg_ratio = round(total_pos/total_neg, 2)
        print(f'{len(self.data)} pairs added with {total_pos} positive pairs and {total_neg} negative pairs | +/- ratio = {self.pos_neg_ratio} | {zero_ref_papers} papers did have references stored in metadata dict.')

    def is_published_after(self, year1, year2):
        return int(year1 < year2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paper_id1, paper_id2, score = self.data[idx]
        embedding1 = self.embedding_map[paper_id1]
        embedding2 = self.embedding_map[paper_id2]
        year1 = self.metadata[paper_id1]['year']
        year2 = self.metadata[paper_id2]['year']
        is_after = self.is_published_after(year1, year2)
        return embedding1, embedding2, torch.tensor(score, dtype=torch.float32), torch.tensor(is_after, dtype=torch.float32)
