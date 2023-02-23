import asyncio
import time
from typing import List, Iterable
from config import S2_API_KEY, S2_RATE_LIMIT
# import httpx  # https://github.com/encode/httpx
import aiohttp
import logging
from pymongo import MongoClient
# from motor.motor_asyncio import AsyncIOMotorClient
import requests


LOG_FILE = 'logs/crawler.log'


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    def __init__(
            self,
            client: aiohttp.ClientSession,
            initial_papers: List[str],  # intial paper IDs to start crawling from
            workers: int = 10,
            max_papers: int = 100,
            s2_rate_limit: int = 20,
            mongo_url: str = 'mongodb://localhost:27017',
    ) -> None:
        self.client = client
        # self.mongo_client = mongo_client
        self.mongo_url = mongo_url
        self.init_db()
        # self.start_urls = set(urls)
        self.initial_papers = initial_papers
        self.todo = asyncio.Queue()
        self.seen = set()
        self.done = set()
        self.s2_rate_limit = s2_rate_limit
        # self.filter_url = filter_url
        self.num_workers = workers
        self.max_papers = max_papers
        self.total = 0
        self.headers = {
            'Content-type': 'application/json',
            'x-api-key': S2_API_KEY
        }

        # self.semaphore = asyncio.Semaphore(s2_rate_limit)

    @classmethod
    def from_dict(cls, settings: dict) -> 'Crawler':
        return cls(**settings)

    def init_db(self) -> None:
        client = MongoClient(self.mongo_url)
        # client = AsyncIOMotorClient(self.mongo_url)
        collection_name = 'async_crawler'
        self.db = client['refpred']
        all_collections = self.db.list_collection_names()
        if collection_name in all_collections:
            logger.info(f"Dropped pre-existing '{collection_name}' collection")
            self.db.drop_collection(collection_name)
        test_collection = self.db[collection_name]
        logger.info(f"Created '{collection_name}' collection")
        self.collection = test_collection

    def get_reference_url(self, paper_id, is_arxiv: bool = False) -> str:
        if is_arxiv:
            return f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}/references?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'
        return f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'

    def get_paper_url(self, paper_id, is_arxiv: bool = False) -> str:
        if is_arxiv:
            return f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'
        return f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'

    async def run(self) -> None:
        initial_paper_id = self.initial_papers[0]
        initial_url = self.get_paper_url(initial_paper_id)
        response = requests.get(initial_url, headers=self.headers)
        if response.status_code != 200:
            logger.exception(f"Error fetching paper {initial_paper_id}")
            return None
        logger.info(f"Fetching intial paper {initial_paper_id}")
        result_data = response.json()
        # result_data['_id'] = result_data['paperId']
        await self.on_found_papers([result_data], initial=True)  # prime the queue
        workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.num_workers)
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
        # cur_paper is a dict
        cur_paper = await self.todo.get()
        try:
            await self.crawl(cur_paper)
        except Exception as exc:
            # retry handling here...
            logger.exception(f"Error processing {cur_paper['_id']}: {exc}")
        finally:
            self.todo.task_done()

    async def crawl(self, cur_paper: dict) -> None:
        # rate limiting to 100 requests / second
        await asyncio.sleep(1/self.num_workers)
        # await asyncio.sleep(0.1)
        cur_paper_id = cur_paper['paperId']
        ref_url = self.get_reference_url(cur_paper_id)
        # cur_paper['_id'] = cur_paper_id
        # response = await self.client.get(cur_paper, follow_redirects=True)
        # async with self.semaphore:
        async with self.client.get(ref_url, headers=self.headers) as response:
            # if self.semaphore.locked():
            #     logger.warning(f"Semaphore locked for {cur_paper_id}")
            #     await asyncio.sleep(1)
            if response.status != 200:
                # logger.exception(
                #     f"Error fetching references for {cur_paper_id} - {response.status}")
                # raise Exception(f"Error fetching references for {cur_paper_id} - {response.status}")
                raise aiohttp.web.HTTPException(
                    f"Error fetching references for {cur_paper_id} - {response.status}")
            logger.info(f"Fetching references for {cur_paper_id} - {response.status}")
            # else:
            #     logger.info(f"Found references for {cur_paper_id}")
            result_data = await response.json()
            found_references = result_data['data']
            # cur_paper['references'] = found_references
            found_references = [ref['citedPaper'] for ref in found_references]
            ref_ids = [ref['paperId'] for ref in found_references if ref['paperId'] is not None]
            cur_paper['references'] = ref_ids
            cur_paper['allReferencesStored'] = True
            if len(ref_ids) != cur_paper['referenceCount']:
                cur_paper['allReferencesStored'] = False
            self.collection.insert_one(cur_paper)

        # found_references = await self.get_paper_references(
        #     base=str(response.url),
        #     text=response.text,
        # )

        self.done.add(cur_paper['paperId'])
        await self.on_found_papers(found_references)

    # async def get_paper_references(self, base: str, text: str) -> set[str]:
    #     parser = UrlParser(base, self.filter_url)
    #     parser.feed(text)
    #     return parser.found_references

    async def on_found_papers(self, papers: List[dict], initial: bool = False) -> None:
        # print(papers)
        if initial:
            for paper in papers:
                await self.put_todo(paper)
            return
        ids = {paper['paperId'] for paper in papers if paper['paperId'] is not None}
        new = ids - self.seen
        self.seen.update(new)

        # await save to database or file here...

        for paper in papers:
            if paper['paperId'] in new:
                await self.put_todo(paper)

    async def put_todo(self, paper: dict) -> None:
        # paper is a dict with fields like paper_id, title, abstract, etc.
        if self.total >= self.max_papers:
            # if len(self.done) >= self.max_papers:
            # logger.info(f"Current todo queue size: {self.todo.qsize()}")
            # logger.info(f"Current done set size: {len(self.done)}")
            return
        self.total += 1
        await self.todo.put(paper)


async def main() -> None:
    # filterer = UrlFilterer(
    #     allowed_domains={"mcoding.io"},
    #     allowed_schemes={"http", "https"},
    #     allowed_filetypes={".html", ".php", ""},
    # )

    start = time.perf_counter()
    async with aiohttp.ClientSession() as client:
        # starting with the famous paper 'Attention is all you need'
        # https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776
        crawler = Crawler(
            client=client,
            initial_papers=["204e3073870fae3d05bcbc2f6a8e263d9b72e776"],
            # filter_url=filterer.filter_url,
            workers=100,
            max_papers=10000,
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

    # seen = sorted(crawler.seen)
    print("Results:")
    # for paper in seen:
    #     print(paper)
    print(f"Crawled: {len(crawler.done)} Papers")
    print(f"Found: {len(crawler.seen)} Papers")
    print(f"Done in {end - start:.2f}s")

    # code to check which of seen are not in database
    # for id, paper_id in enumerate(crawler.done, 1):
    #     if crawler.collection.find_one({'_id': paper_id}):
    #         print(f"{id}. Paper {paper_id} in database")
    #     else:
    #         print(f"{id}. Paper {paper_id} not in database")


if __name__ == '__main__':
    asyncio.run(main())
