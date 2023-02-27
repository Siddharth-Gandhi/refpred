import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List

import httpx  # https://github.com/encode/httpx

# from motor.motor_asyncio import AsyncIOMotorClient
import requests
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config import S2_API_KEY, S2_RATE_LIMIT

# import aiohttp


LOG_FILE = "logs/crawler.log"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# async def create_async_crawler(settings):
#     crawler = Crawler.from_dict(settings)
#     await crawler.init_db()
#     return crawler


class RateLimitExceededException(Exception):
    """Exception raised when rate limit is exceeded"""

    pass


class TimeoutException(Exception):
    """Exception raised when a request times out"""

    pass


@dataclass(frozen=True)
class MongoDB:
    """A MongoDB database"""

    mongo_url: str = "mongodb://localhost:27017"
    db_name: str = "refpred"
    collection_name: str = "test"
    client: MongoClient = field(init=False, repr=False)
    db: Database[Any] = field(init=False, repr=False)
    collection: Collection[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the MongoDB database and collection"""
        self.client = MongoClient(self.mongo_url)
        self.db = self.client[self.db_name]
        all_collections = self.db.list_collection_names()
        if self.collection_name in all_collections:
            logger.warning(f"Dropped pre-existing '{self.collection_name}' collection")
            self.db.drop_collection(self.collection_name)
        logger.info(f"Created '{self.collection_name}' collection")
        self.collection = self.db[self.collection_name]


@dataclass
class Crawler:
    """A crawler for the Semantic Scholar API"""

    client: httpx.AsyncClient
    initial_papers: List[str]
    workers: int = 10
    max_papers: int = 100
    todo = asyncio.Queue()
    seen = set()
    done = set()
    retry = set()
    total = 0
    stored = 0
    headers = {
        "Content-type": "application/json",
        "x-api-key": S2_API_KEY,
    }
    mongodb: MongoDB = MongoDB()

    @classmethod
    def from_dict(cls, settings: dict) -> "Crawler":
        """
        Create a Crawler instance from a dict of settings"""
        return cls(**settings)

    def init_db(self) -> None:
        """Initialize the MongoDB database and collection"""
        client = MongoClient(self.mongo_url)
        # collection_name = 'async_crawler'
        self.db = client[self.db_name]
        all_collections = self.db.list_collection_names()
        if self.collection_name in all_collections:
            logger.warning(f"Dropped pre-existing '{self.collection_name}' collection")
            self.db.drop_collection(self.collection_name)
        test_collection = self.db[self.collection_name]
        logger.info(f"Created '{self.collection_name}' collection")
        self.collection = test_collection

    def get_reference_url(self, paper_id, is_arxiv: bool = False) -> str:
        """Get the URL for the references of a paper"""
        if is_arxiv:
            return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}/references?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"
        return f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"

    def get_paper_url(self, paper_id, is_arxiv: bool = False) -> str:
        """Get the URL for a paper"""
        if is_arxiv:
            return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"
        return f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"

    async def init_queue(self) -> None:
        """Initialize the queue with the initial papers"""
        initial_paper_id = self.initial_papers[0]
        initial_url = self.get_paper_url(initial_paper_id)
        response = requests.get(initial_url, headers=self.headers, timeout=10)
        if response.status_code != 200:
            logger.error(f"Error fetching paper {initial_paper_id}")
            return None
        logger.debug(f"Fetching intial paper {initial_paper_id}")
        result_data = response.json()
        result_data["_id"] = result_data["paperId"]
        # prime the queue
        await self.on_found_papers([result_data], initial=True)

    async def run(self) -> None:
        """Run the crawler by creating workers until todo queue is empty"""
        await self.init_queue()
        workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]
        await self.todo.join()
        for worker in workers:
            worker.cancel()

    async def worker(self) -> None:
        """One worker processes one paper at a time from the queue in a loop until cancelled"""
        while True:
            try:
                await self.process_one()
            except asyncio.CancelledError:
                return

    async def retry_crawl(self, paper):
        """Retry crawling a paper in case of an exception"""
        if paper["_id"] in self.retry:
            logger.error(f"Error processing {paper['_id']} even after retrying")
            return
        logger.info(f"Retrying for {paper['_id']}")
        self.retry.add(paper["_id"])
        # await self.todo.put_nowait(cur_paper)
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
        await asyncio.sleep(0.5)

        cur_paper_id = cur_paper["paperId"]
        ref_url = self.get_reference_url(cur_paper_id)
        cur_paper["_id"] = cur_paper_id
        if cur_paper["title"] is None or cur_paper["abstract"] is None:
            logger.debug(f"Skipping {cur_paper_id} as empty title or abstract")
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

        ref_ids = [ref["paperId"] for ref in found_references if ref["paperId"] is not None]
        cur_paper["references"] = ref_ids
        cur_paper["allReferencesStored"] = True
        if len(ref_ids) != cur_paper["referenceCount"]:
            cur_paper["allReferencesStored"] = False

        self.collection.insert_one(cur_paper)
        self.done.add(cur_paper["paperId"])
        self.stored += 1
        if self.stored % 100 == 0:
            logger.info(f"Stored {self.stored} papers")

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
    # timeout_sec = 100
    # my_timeout = aiohttp.ClientTimeout(total=None)
    # client_args = dict(
    #     timeout=my_timeout,
    #     trust_env=True,
    # )
    timeout = httpx.Timeout(10, connect=10, read=None, write=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # starting with the famous paper 'Attention is all you need'
        # https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776
        crawler = Crawler(
            client=client,
            initial_papers=["204e3073870fae3d05bcbc2f6a8e263d9b72e776"],
            # filter_url=filterer.filter_url,
            workers=S2_RATE_LIMIT,
            max_papers=10000,
            db_name="refpred",
            collection_name="async_crawler_test2",
        )
        # settings = {
        #     'client': client,
        #     'initial_papers': ["204e3073870fae3d05bcbc2f6a8e263d9b72e776"],
        #     'workers': 100,
        #     'max_papers': 100,
        # }
        # crawler = await create_async_crawler(settings)
        # time.sleep(1)
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
