import asyncio
import json

# import numpy as np
# import pandas as pd
import logging
import time

import aiohttp
import requests
from pymongo import MongoClient

from config import S2_API_KEY, S2_RATE_LIMIT
from data.old_db import get_papers_db

pdb = get_papers_db(flag='r')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# print(next(papers))
base_url = 'https://api.semanticscholar.org/graph/v1'
# batch_url = 'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,referenceCount,citationCount,authors,externalIds'

batch_url = 'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'
headers = {
    'Content-type': 'application/json',
    'x-api-key': S2_API_KEY
}


def get_papers():
    keys = pdb.keys()
    for p in keys:
        d = pdb[p]
        author_str = ', '.join([a['name'] for a in d['authors']])
        yield {
            'arxiv_id': d['_id'],
            'title': d['title'],
            'authors': author_str,
            'abstract': d['summary'],
        }


def arxiv_to_s2id(arxiv_id):
    s2_search = f"{base_url}/paper/arXiv:{arxiv_id}"
    logger.debug(f"Searching Semantic Scholar for {s2_search}")
    response = requests.get(s2_search, headers=headers)
    if response.status_code != 200:
        logger.debug("Semantic Scholar did not return status 200 response")
        return None
        # print(response.json())
    s2_response = response.json()
    # return S2 ID
    return s2_response['paperId']


def get_reference_url(paper_id, is_arxiv=False):
    if is_arxiv:
        return f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}/references?fields=title,abstract'
    return f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,abstract'


def add_to_db(obj_list, db_name='refpred', collection_name='test'):
    client = MongoClient("mongodb://localhost:27017")
    db = client['refpred']

    # drop 'test' collection if it exists
    if collection_name in db.list_collection_names():
        logger.info(f"Dropped pre-existing '{collection_name}' collection")
        db.drop_collection(collection_name)

    time.sleep(3)
    # create new 'test' collection
    test_collection = db[collection_name]
    logger.info(f"Created '{collection_name}' collection")
    test_collection.insert_many(obj_list)
    logger.info(f"Added {len(obj_list)} papers to '{collection_name}' collection")


async def add_refs_to_paper_obj(session, paper_object, papers, semaphore):
    arxiv_id = paper_object['externalIds']['ArXiv']
    url = get_reference_url(arxiv_id, is_arxiv=True)
    paper_object['_id'] = paper_object['paperId']
    for paper in papers:
        if paper['arxiv_id'] == arxiv_id:
            paper_object['abstract'] = paper['abstract']
    async with semaphore:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                logger.error(f"Error fetching references for {arxiv_id}")
                return paper_object
            logger.info(f"Fetching references for {arxiv_id}")
            result_data = await response.json()
            reference_papers = result_data['data']
        # reference_papers = requests.get(url, headers=headers).json()['data']
            paper_object['references'] = reference_papers
            return paper_object


async def main(batch_response, papers):
    semaphore = asyncio.Semaphore(S2_RATE_LIMIT)  # allow 5 requests at a  time
    async with aiohttp.ClientSession() as session:
        tasks = []
        for paper_obj in batch_response:
            task = asyncio.ensure_future(add_refs_to_paper_obj(
                session, paper_obj, papers, semaphore))
            tasks.append(task)
        paper_list = await asyncio.gather(*tasks)

    add_to_db(paper_list, collection_name='test2')
    # print(paper_list)


if __name__ == '__main__':
    ids = []
    papers = get_papers()
    for i, paper in enumerate(papers, start=1):
        # print(f"{i}. {paper['arxiv_id']} - {paper['title']}")
        id_str = f"ARXIV:{paper['arxiv_id']}"
        ids.append(id_str)

    data = json.dumps({"ids": ids})
    response = requests.post(batch_url, data=data, headers=headers)
    print(response.status_code)
    batch_response = response.json()
    asyncio.run(main(batch_response, papers))
