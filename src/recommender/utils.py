import torch


@torch.no_grad()
def get_embedding(paper, specter_tokenizer, specter_model, embedding_map=None):
    '''Paper is a dict with keys: paper_id, title (required), abstract'''
    if embedding_map and paper['paper_id'] is not None and paper['paper_id'] in embedding_map:
        return torch.tensor(embedding_map[paper['paper_id']]).view(1,-1)
    # tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    # model = AutoModel.from_pretrained('allenai/specter')
    title_abs = [d['title'] + specter_tokenizer.sep_token + (d.get('abstract') or '') for d in [paper]]
    inputs = specter_tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = specter_model(**inputs)
    cur_embedding = result.last_hidden_state[:, 0, :]
    return cur_embedding