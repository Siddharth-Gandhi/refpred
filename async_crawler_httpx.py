import asyncio
import logging
import time
from typing import List

import httpx  # https://github.com/encode/httpx
# from motor.motor_asyncio import AsyncIOMotorClient
import requests
from pymongo import MongoClient

from config import S2_API_KEY, S2_RATE_LIMIT

# import aiohttp


LOG_FILE = "logs/crawler.log"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# async def create_async_crawler(settings):
#     crawler = Crawler.from_dict(settings)
#     await crawler.init_db()
#     return crawler


class Crawler:
    """A crawler for the Semantic Scholar API"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        # intial paper IDs to start crawling from
        initial_papers: List[str],
        workers: int = 10,
        max_papers: int = 100,
        # s2_rate_limit: int = 20,
        mongo_url: str = "mongodb://localhost:27017",
        db_name: str = "refpred",
        collection_name: str = "test",
    ) -> None:
        self.client = client
        self.mongo_url = mongo_url
        self.initial_papers = initial_papers
        self.todo = asyncio.Queue()
        self.seen = set()
        self.done = set()
        self.retry = set()
        # self.s2_rate_limit = s2_rate_limit
        self.num_workers = workers
        self.max_papers = max_papers
        self.total = 0
        self.stored = 0
        self.headers = {
            "Content-type": "application/json",
            "x-api-key": S2_API_KEY,
        }
        self.db_name = db_name
        self.collection_name = collection_name
        # self.semaphore = asyncio.Semaphore(s2_rate_limit)
        self.init_db()

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
            logger.warning(
                f"Dropped pre-existing '{self.collection_name}' collection"
            )
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
        """Run the crawler"""
        await self.init_queue()
        workers = [
            asyncio.create_task(self.worker()) for _ in range(self.num_workers)
        ]
        await self.todo.join()
        for worker in workers:
            worker.cancel()

    async def worker(self) -> None:
        while True:
            try:
                await self.process_one()
            except asyncio.CancelledError:
                return

    async def process_one(self) -> None:
        """Gets one paper from the queue and processes it"""
        # cur_paper is a dict
        cur_paper = await self.todo.get()
        try:
            await self.crawl(cur_paper)
        except Exception as exc:
            if cur_paper["_id"] in self.retry:
                logger.error(
                    f"Error processing {cur_paper['_id']}: {exc.__class__.__name__}"
                )
                return
            logger.error(
                f"Retrying for {cur_paper['_id']} - {exc.__class__.__name__}"
            )
            self.retry.add(cur_paper["_id"])
            # await self.todo.put_nowait(cur_paper)
            await asyncio.sleep(0.5)
            await self.crawl(cur_paper)
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
        # async with self.semaphore:
        # async with self.client.get(ref_url, headers=self.headers) as response:

        # timeout = httpx.Timeout(10.0, read_timeout=None)

        response = await self.client.get(ref_url, headers=self.headers)

        # if self.semaphore.locked():
        #     logger.warning(f"Semaphore locked for {cur_paper_id}")
        #     await asyncio.sleep(1)

        if response.status_code == 429:
            logger.critical(
                f"Rated limited for {cur_paper_id} - {response.status_code}"
            )
            # await self.todo.put_nowait(cur_paper)
            await asyncio.sleep(1)
            await self.crawl(cur_paper)
            return

        if response.status_code != 200:
            logger.error(
                f"Error fetching references for {cur_paper_id} - {response.status_code}"
            )
            # raise specific exception
            # return
            raise asyncio.exceptions.TimeoutError(
                f"Error fetching references for {cur_paper_id} - {response.status_code}"
            )

        logger.debug(
            f"Fetching references for {cur_paper_id} - {response.status_code}"
        )

        result_data = response.json()
        found_references = result_data["data"]
        found_references = [ref["citedPaper"] for ref in found_references]

        ref_ids = [
            ref["paperId"]
            for ref in found_references
            if ref["paperId"] is not None
        ]
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

    async def on_found_papers(
        self, papers: List[dict], initial: bool = False
    ) -> None:
        """
      Called when new papers are found. Filters out papers that have already been seen and puts the new ones in the queue.
        """
        if initial:
            for paper in papers:
                await self.put_todo(paper)
            return
        ids = {
            paper["paperId"]
            for paper in papers
            if paper["paperId"] is not None
        }
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
    async with httpx.AsyncClient(timeout=100) as client:
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

    print("Results:")
    print(f"Crawled: {len(crawler.done)} Papers")
    print(f"Found: {len(crawler.seen)} Papers")
    print(f"Done in {end - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
