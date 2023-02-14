from config import S2_API_KEY
import logging
import requests
from pymongo import MongoClient


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# logging.basicConfig(
#     level=logging.INFO, format='%(name)s %(levelname)s %(asctime)s %(message)s',
#     datefmt='%m/%d/%Y %I:%M:%S %p')


client = MongoClient("mongodb://localhost:27017")
db = client['refpred']

# drop 'test' collection if it exists
if 'test' in db.list_collection_names():
    db.drop_collection('test')

# create new 'test' collection
test_collection = db['test']


INITIAL_PAPER = '1706.03762'
base_url = 'https://api.semanticscholar.org/graph/v1'

headers = {
    'Content-type': 'application/json',
    'x-api-key': S2_API_KEY
}


def get_reference_url(paper_id, is_arxiv=False):
    if is_arxiv:
        return f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}/references?fields=title,abstract'
    return f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,abstract'


def arxiv_to_s2id(arxiv_id):
    s2_search = f"{base_url}/paper/arXiv:{arxiv_id}"
    logger.debug(f"Searching Semantic Scholar for {s2_search}")
    response = requests.get(s2_search, headers=headers)
    if response.status_code != 200:
        logger.error(f"Semantic Scholar did not return status 200 response")
        return None
        # print(response.json())
    s2_response = response.json()
    s2_id = s2_response['paperId']
    return s2_id


def get_references(s2_id):
    # print('hi')
    s2_search = get_reference_url(s2_id)
    logger.debug(f"Searching Semantic Scholar for {s2_search}")
    response = requests.get(s2_search, headers=headers)
    if response.status_code != 200:
        logger.error(f"Semantic Scholar did not return status 200 response")
        return None
    s2_response = response.json()
    return s2_response


if __name__ == '__main__':
    # print('hi')
    s2_id = arxiv_to_s2id(INITIAL_PAPER)
    logger.info(f"Semantic Scholar ID: {s2_id}\n")
    references = get_references(s2_id)
    logging.debug(references)
