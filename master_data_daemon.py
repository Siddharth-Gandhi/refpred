# import argparse
# from random import shuffle
from config import S2_API_KEY
# import numpy as np
import logging
from data.db import get_papers_db, save_features
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    pdb = get_papers_db(flag='r')

    def get_papers():
        keys = pdb.keys()
        for p in keys:
            d = pdb[p]
            author_str = ', '.join([a['name'] for a in d['authors']])
            # filtered_dict = {k: d[k] for k in ['_id', 'title']}
            filtered_dict = {}
            filtered_dict['arxiv_id'] = d['_id']
            filtered_dict['title'] = d['title']
            filtered_dict['authors'] = author_str
            filtered_dict['abstract'] = d['summary']
            yield filtered_dict

    papers = get_papers()
    # print(next(papers))
    base_url = 'https://api.semanticscholar.org/graph/v1'

    headers = {
        'Content-type': 'application/json',
        'x-api-key': S2_API_KEY
    }

    def arxiv_to_s2id(arxiv_id):
        s2_search = f"{base_url}/paper/arXiv:{arxiv_id}"
        logger.debug(f"Searching Semantic Scholar for {s2_search}")
        response = requests.get(s2_search, headers=headers)
        if response.status_code != 200:
            logger.debug(f"Semantic Scholar did not return status 200 response")
            return None
            # print(response.json())
        s2_response = response.json()
        s2_id = s2_response['paperId']
        return s2_id

    for paper in papers:
        # print(f"{paper['arxiv_id']} - {paper['title']}")
        s2_id = arxiv_to_s2id(paper['arxiv_id'])
        # print(f"Semantic Scholar ID: {s2_id}\n")
        if s2_id is not None:
            paper['s2_id'] = s2_id

    # batch_url = 'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'

    batch_url = 'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,referenceCount,citationCount,authors,externalIds'
    response = requests.post(batch_url, data=data, headers=headers)
    json_response = response.json()
