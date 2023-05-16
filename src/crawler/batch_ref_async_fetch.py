import asyncio
import json
import os
import sys
import time
from contextlib import AbstractAsyncContextManager

import httpx
import numpy as np
import pandas as pd
import requests
from db import MongoDBClient
from tqdm.asyncio import tqdm


class TqdmAsync(AbstractAsyncContextManager):
    def __init__(self, *args, **kwargs):
        self._tqdm = tqdm(*args, **kwargs)

    def update(self, n):
        self._tqdm.update(n)

    async def __aenter__(self):
        return self._tqdm

    async def __aexit__(self, exc_type, exc_value, traceback):
        self._tqdm.close()



def get_batch_url() -> str:
    """Get the URL for a batch of papers"""
    # return 'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'

    return 'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,year'

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# async def fetch_and_store_data(client, ids_batch, mongodb_client, progress):
#     # Replace this URL with the actual API endpoint.
#     url = "https://api.semanticscholar.org/graph/v1/paper/batch"
#     params={'fields': 'abstract,title,year'}

#     # Modify the payload if necessary, depending on the API's requirements.
#     payload = {"ids": ids_batch}
#     try:
#         response = await client.post(url, json=payload, params=params, headers=headers, timeout=40)
#     except httpx.TimeoutException:
#         print(f"Request timed out for IDs: {ids_batch}")
#         return


#     if response.status_code == 429:
#         # rate limit
#         print(f"Rate limit reached. Waiting for {response.headers['Retry-After']} seconds")
#         time.sleep(int(response.headers['Retry-After']))
#         return await fetch_and_store_data(client, ids_batch, mongodb_client, progress)

#     if response.status_code != 200:
#         print(f"Error fetching data for IDs: {ids_batch}")
#         return
#     result_data = response.json()

#     data = response.json()
#     for paper in result_data:
#         paper["_id"] = paper["paperId"]
#     # Replace this with the actual collection in your MongoClient instance.
#     mongodb_client.insert_many(data)

#     print(f"Stored data for {len(data)} papers")
#     # Sleep for a second to avoid rate limiting.
#     await asyncio.sleep(1)

#     # Update the progress bar.
#     progress.update(1)

async def fetch_and_store_data(client, ids_batch, mongodb_client, max_retries=3):
    # Replace this URL with the actual API endpoint.
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params={'fields': 'abstract,title,year'}

    # Modify the payload if necessary, depending on the API's requirements.
    payload = {"ids": ids_batch}

    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, json=payload, params=params, headers=headers, timeout=40)
        except httpx.TimeoutException:
            if attempt < max_retries:
                await asyncio.sleep(5)  # Sleep before retrying.
                continue
            else:
                print(f"Request timed out for IDs: {ids_batch}")
                return

        if response.status_code == 429:
            # rate limit
            print(f"Rate limit reached. Waiting for {response.headers['Retry-After']} seconds")
            await asyncio.sleep(int(response.headers['Retry-After']))
            continue

        if response.status_code != 200:
            if attempt < max_retries:
                print(f"Error in attempt {attempt+1} for  fetchin data for {len(ids_batch)} IDs. [{response.status_code}] Retrying...")
                await asyncio.sleep(1)  # Sleep before retrying.
                continue
            else:
                print(f"Error fetching data for IDs: {len(ids_batch)} with status code {response.status_code}")
                return

        result_data = response.json()
        # Filter out None objects from result_data
        result_data = [paper for paper in result_data if paper is not None]


        for paper in result_data:
            paper["_id"] = paper["paperId"]

        # Replace this with the actual collection in your MongoClient instance.
        mongodb_client.insert_many(result_data)

        # print(f"Stored data for {len(result_data)} papers")
        break

    # Sleep for a second to avoid rate limiting.
    await asyncio.sleep(1)

    # Update the progress bar.
    # progress.update(1)




async def worker(client, ids_batches, mongodb_client, max_retries):
    while ids_batches:
        ids_batch = ids_batches.pop(0)
        await fetch_and_store_data(client, ids_batch, mongodb_client, max_retries=max_retries)


async def fetch_and_store_data_concurrently(ids_batches, mongodb_client, num_workers=10, max_retries=3):
    async with httpx.AsyncClient() as client:
        tasks = []
        # async with TqdmAsync(total=len(ids_batches), desc="Processing batches") as progress:
        for _ in range(num_workers):
            task = asyncio.create_task(worker(client, ids_batches, mongodb_client, max_retries))
            tasks.append(task)
        await asyncio.gather(*tasks)




if __name__ == '__main__':
    sys.path.append(os.getcwd())
    from config import S2_API_KEY  # pylint: disable=import-error
    headers={
        "Content-type": "application/json",
        "x-api-key": S2_API_KEY,
    }
    batch_url = get_batch_url()
    mongodb_client = MongoDBClient(mongo_url='mongodb://localhost:27017', db_name='refpred', collection_name='all_references', init_new=False)

    existing_ids = set(mongodb_client.get_ids())
    print(f"Total existing IDs: {len(existing_ids)}")

    # read ref_ids from data/all_ref_ids.txt where each id is on a separate line

    ref_ids = []
    with open('data/all_ref_ids.txt', 'r') as f:
        ref_ids.extend(line.strip() for line in f)

    ref_ids = [ref_id for ref_id in ref_ids if ref_id not in existing_ids]

    chunks = split_list(ref_ids, 100)

    # chunks = chunks[:100]

    print(f"Total chunks: {len(chunks)}")
    asyncio.run(fetch_and_store_data_concurrently(chunks, mongodb_client, num_workers=10, max_retries=5))

