'''
A crawler for the Semantic Scholar API.
'''

import asyncio
import json
import logging
import logging.config
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set

import httpx  # https://github.com/encode/httpx
import requests

from config import S2_API_KEY, S2_RATE_LIMIT
from db import MongoDBClient
from utils import get_batch_url, get_reference_url

logging.config.fileConfig(fname="logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class RateLimitExceededException(Exception):
    """Exception raised when rate limit is exceeded"""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"RateLimitExceededException: {self.message}"


class TimeoutException(Exception):
    """Exception raised when a request times out"""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"TimeoutException: {self.message}"


@dataclass()
class Crawler:
    """A crawler for the Semantic Scholar API"""

    client: httpx.AsyncClient = field(repr=False)
    initial_papers: List[str] = field(default_factory=list)
    num_workers: int = 10
    max_papers: int = 100
    mongodb_client: MongoDBClient = field(default_factory=MongoDBClient)
    headers: dict = field(repr=False, default_factory=dict)
    todo: asyncio.Queue = field(init=False, repr=False, default_factory=asyncio.Queue)
    seen: Set[str]= field(init=False, default_factory=set)
    done: Set[str] = field(init=False, default_factory=set)
    retry: Dict[str, int] = field(init=False, default_factory=dict)
    total: int = field(init=False, default=0)
    MAX_RETRIES: int = field(init=False, default=3)

    @classmethod
    def from_dict(cls, settings: dict) -> "Crawler":
        """
        Create a Crawler instance from a dict of settings"""
        return cls(**settings)

    async def run(self) -> None:
        """Run the crawler by creating workers until todo queue is empty"""
        self.init_done()
        await self.init_queue()
        workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]
        await self.todo.join()
        for worker in workers:
            worker.cancel()

    async def init_queue(self) -> None:
        """Initialize the queue with the initial papers"""
        batch_url = get_batch_url()
        data = json.dumps({"ids": self.initial_papers})
        response = requests.post(url=batch_url, data=data, headers=self.headers, timeout=10)
        # initial_paper_id = self.initial_papers[0]
        # initial_url = get_paper_url(initial_paper_id)
        # response = requests.get(initial_url, headers=self.headers, timeout=10)
        if response.status_code != 200:
            logger.error("Error fetching initial papers")
            sys.exit(1)
        logger.debug(f"Fetching data for intial papers {self.initial_papers}")
        result_data = response.json()
        # result_data["_id"] = result_data["paperId"]
        for paper in result_data:
            paper["_id"] = paper["paperId"]
        # prime the queue
        await self.on_found_papers(result_data, initial=True)

    def init_done(self) -> None:
        """Initialize the seen set with already stored papers from DB"""
        # self.seen = set(self.initial_papers)
        self.done = self.mongodb_client.get_ids()
        logger.info(f"Already stored {len(self.done)} papers")

    async def worker(self) -> None:
        """One worker processes one paper at a time from the queue in a loop until cancelled"""
        while True:
            try:
                await self.process_one()
            except asyncio.CancelledError:
                return

    async def retry_crawl(self, paper) -> None:
        """Retry crawling a paper in case of an exception"""
        if paper["_id"] in self.retry and self.retry[paper["_id"]] > self.MAX_RETRIES:
            logger.error(f"Error processing {paper['_id']} even after retrying {self.MAX_RETRIES} times")
            return
        # self.retry.add(paper["_id"])
        self.retry[paper["_id"]] = self.retry.get(paper["_id"], 0) + 1
        logger.info(f"Retry #{self.retry[paper['_id']]} for {paper['_id']}")
        # await self.todo.put_nowait(cur_paper)
        await asyncio.sleep(1)
        await self.crawl(paper)

    async def process_one(self) -> None:
        """Gets one paper from the queue and processes it"""
        # cur_paper is a dict
        cur_paper = await self.todo.get()
        try:
            await self.crawl(cur_paper)
        except TimeoutException as te:
            # logger.warning(f"Timeout for {cur_paper['_id']}")
            logger.warning(te)
            await self.retry_crawl(cur_paper)
        except RateLimitExceededException as rlee:
            logger.critical("Rate limit exceeded, retrying in 2 second")
            logger.critical(rlee)
            await asyncio.sleep(2)
            await self.retry_crawl(cur_paper)
        finally:
            self.todo.task_done()

    async def crawl(self, cur_paper: dict) -> None:
        """
        Crawl a paper and its references, stores them in the database.
        """
        # TODO proper rate limiting to 100 requests / second
        # await asyncio.sleep(1 / self.num_workers)
        await asyncio.sleep(1)

        cur_paper_id = cur_paper["paperId"]
        ref_url = get_reference_url(cur_paper_id)
        cur_paper["_id"] = cur_paper_id
        if cur_paper["title"] is None or cur_paper["abstract"] is None:
            logger.debug(f"Skipping {cur_paper_id} as empty title or abstract")
            # I have no clue why this total -= 1 is here, it shouldn't be required, but crawler just prematurely stops
            self.total -= 1
            return
        # async with self.semaphore:
        # async with self.client.get(ref_url, headers=self.headers) as response:

        response = await self.client.get(ref_url, headers=self.headers)

        # if self.semaphore.locked():
        #     logger.warning(f"Semaphore locked for {cur_paper_id}")
        #     await asyncio.sleep(1)

        if response.status_code == 429:
            # logger.critical(
            #     f"Rated limited for {cur_paper_id} - {response.status_code}"
            # )
            # # await self.todo.put_nowait(cur_paper)
            # await asyncio.sleep(1)
            # await self.crawl(cur_paper)
            raise RateLimitExceededException(
                f"Rated limited for {cur_paper_id} - {response.status_code}"
            )

        if response.status_code == 504:
            # raise asyncio.exceptions.TimeoutError(
            #     f"Timeout for {cur_paper_id} - {response.status_code}"
            # )
            raise TimeoutException(f"Timeout for {cur_paper_id} - {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Error fetching references for {cur_paper_id} - {response.status_code}")
            return

        logger.debug(f"Fetching references for {cur_paper_id} - {response.status_code}")

        result_data = response.json()
        found_references = result_data["data"]
        found_references = [ref["citedPaper"] for ref in found_references]
        found_references = sorted(found_references, key=lambda x: x["citationCount"] or 0, reverse=True)
        ref_ids = [ref["paperId"] for ref in found_references if ref["paperId"] is not None]
        cur_paper["references"] = ref_ids
        cur_paper["allReferencesStored"] = True
        if len(ref_ids) != cur_paper["referenceCount"]:
            cur_paper["allReferencesStored"] = False

        # self.collection.insert_one(cur_paper)
        self.mongodb_client.insert_one(cur_paper)
        self.done.add(cur_paper["paperId"])
        # self.stored += 1
        # if self.stored % 100 == 0:
        #     logger.info(f"Stored {self.stored} papers")

        await self.on_found_papers(found_references)

    # async def get_paper_references(self, base: str, text: str) -> set[str]:
    #     parser = UrlParser(base, self.filter_url)
    #     parser.feed(text)
    #     return parser.found_references

    async def on_found_papers(self, papers: List[dict], initial: bool = False) -> None:
        """
        Called when new papers are found. Filters out papers that have already been seen and puts the new ones in the queue.
        """
        if initial:
            for paper in papers:
                await self.put_todo(paper)
            return
        ids = {paper["paperId"] for paper in papers if paper["paperId"] is not None}
        new = ids - self.seen
        self.seen.update(new)

        for paper in papers:
            if paper["paperId"] in new:
                await self.put_todo(paper)

    async def put_todo(self, paper: dict) -> None:
        """Put a paper in the queue"""
        # paper is a dict with fields like paper_id, title, abstract, etc.
        if self.total >= self.max_papers:
            return
        self.total += 1
        await self.todo.put(paper)


async def main() -> None:
    """Main function"""
    start = time.perf_counter()
    headers={
        "Content-type": "application/json",
        "x-api-key": S2_API_KEY,
    }
    mongodb_client = MongoDBClient(mongo_url='mongodb://localhost:27017', db_name='refpred', collection_name='review2_demo', init_new=True)
    timeout = httpx.Timeout(10, connect=10, read=None, write=10)
    # based on https://towardsdatascience.com/top-10-research-papers-in-ai-1f02cf844e26
    initial_papers = ["204e3073870fae3d05bcbc2f6a8e263d9b72e776", "bee044c8e8903fb67523c1f8c105ab4718600cdb", "36eff562f65125511b5dfab68ce7f7a943c27478", "8388f1be26329fa45e5807e968a641ce170ea078", "846aedd869a00c09b40f1f1f35673cb22bc87490", "e0e9a94c4a6ba219e768b4e59f72c18f0a22e23d", "fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5", "424561d8585ff8ebce7d5d07de8dbf7aae5e7270", "4d376d6978dad0374edfa6709c9556b42d3594d3", "a6cb366736791bcccc5c8639de5a8f9636bf87e8", "df2b0e26d0599ce3e70df8a9da02e51594e0e992", "913f54b44dfb9202955fe296cf5586e1105565ea", "156d217b0a911af97fa1b5a71dc909ccef7a8028", "a3e4ceb42cbcd2c807d53aff90a8cb1f5ee3f031", "5c5751d45e298cea054f32b392c12c61027d2fe7", "bc1586a2e74d6d1cf87b083c4cbd1eede2b09ea5", "921b2958cac4138d188fd5047aa12bbcf37ac867", "cb92a7f9d9dbcf9145e32fdfa0e70e2a6b828eb1"]
    MAX_PAPERS = 10000
    async with httpx.AsyncClient(timeout=timeout) as client:
        # starting with the famous paper 'Attention is all you need'
        crawler = Crawler(
            client=client,
            initial_papers=initial_papers,
            num_workers=S2_RATE_LIMIT,
            max_papers=MAX_PAPERS,
            mongodb_client=mongodb_client,
            headers=headers,
        )
        await crawler.run()
    end = time.perf_counter()

    logger.info("Results:")
    logger.info(f"Crawled: {len(crawler.done)} Papers")
    logger.info(f"Found: {len(crawler.seen)} Papers")
    logger.info(f"Done in {end - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())


# TODO
# 1. Batch processing of seed papers
# 2. Initialize seen from dataset to avoid restarting over
# 3. Null abstract papers need to be removed from the dataset ✅
