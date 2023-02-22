import asyncio
# import html.parser
# import pathlib
import time
# import urllib.parse
from typing import List, Iterable
from config import S2_API_KEY, S2_RATE_LIMIT
# import httpx  # https://github.com/encode/httpx
import aiohttp
import logging
from pymongo import MongoClient
import requests
# from aiolimiter import AsyncLimiter

# limiter = AsyncLimiter(S2_RATE_LIMIT)

LOG_FILE = 'logs/crawler.log'


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(LOG_FILE)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


class Crawler:
    def __init__(
            self,
            client: aiohttp.ClientSession,
            initial_papers: List[str],  # intial paper IDs to start crawling from
            workers: int = 10,
            max_papers: int = 100,
            s2_rate_limit: int = 20,
            # mongo_client: MongoClient = None,
            mongo_url: str = 'mongodb://localhost:27017',
    ):
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

    def init_db(self):
        client = MongoClient(self.mongo_url)
        collection_name = 'async_crawler'
        self.db = client['refpred']
        if collection_name in self.db.list_collection_names():
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

    async def run(self):
        initial_paper_id = self.initial_papers[0]
        initial_url = self.get_paper_url(initial_paper_id)
        response = requests.get(initial_url, headers=self.headers)
        if response.status_code != 200:
            logger.exception(f"Error fetching paper {initial_paper_id}")
            return None
        logger.info(f"Fetching intial paper {initial_paper_id}")
        result_data = response.json()
        result_data['_id'] = result_data['paperId']
        await self.on_found_papers([result_data], initial=True)  # prime the queue
        workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.num_workers)
        ]
        await self.todo.join()

        for worker in workers:
            worker.cancel()

    async def worker(self):
        while True:
            try:
                await self.process_one()
            except asyncio.CancelledError:
                return

    async def process_one(self):
        # cur_paper is a dict
        cur_paper = await self.todo.get()
        try:
            await self.crawl(cur_paper)
        except Exception as exc:
            # retry handling here...
            logger.exception(f"Error processing {cur_paper['_id']}: {exc}")
        finally:
            self.todo.task_done()

    async def crawl(self, cur_paper: dict):
        # rate limiting to 100 requests / second
        await asyncio.sleep(1/self.num_workers)
        cur_paper_id = cur_paper['paperId']
        ref_url = self.get_reference_url(cur_paper_id)
        cur_paper['_id'] = cur_paper_id
        # response = await self.client.get(cur_paper, follow_redirects=True)
        # async with self.semaphore:
        logger.info(f"Fetching references for {cur_paper_id}")
        async with self.client.get(ref_url, headers=self.headers) as response:
            # if self.semaphore.locked():
            #     logger.warning(f"Semaphore locked for {cur_paper_id}")
            #     await asyncio.sleep(1)
            if response.status != 200:
                logger.exception(f"Error fetching references for {cur_paper_id}")
                return None
            result_data = await response.json()
            found_references = result_data['data']
            # cur_paper['references'] = found_references
            found_references = [ref['citedPaper'] for ref in found_references]
            ref_ids = [ref['paperId'] for ref in found_references if ref['paperId'] is not None]
            cur_paper['references'] = ref_ids
            cur_paper['allReferencesStored'] = True
            if ref_ids:
                if len(ref_ids) != cur_paper['referenceCount']:
                    cur_paper['allReferencesStored'] = False
                    # logger.warning(f"Reference count mismatch for {cur_paper_id}") TODO uncomment
                self.collection.insert_one(cur_paper)

        # found_references = await self.get_paper_references(
        #     base=str(response.url),
        #     text=response.text,
        # )

        await self.on_found_papers(found_references)

        self.done.add(cur_paper['paperId'])

    # async def get_paper_references(self, base: str, text: str) -> set[str]:
    #     parser = UrlParser(base, self.filter_url)
    #     parser.feed(text)
    #     return parser.found_references

    async def on_found_papers(self, papers: List[dict], initial: bool = False):
        # print(papers)
        if initial:
            for paper in papers:
                await self.put_todo(paper)
            return
        ids = {paper['paperId'] for paper in papers if paper['paperId'] is not None}
        new = ids - self.seen
        self.seen.update(new)  # TODO review and maybe uncomment this

        # await save to database or file here...

        for paper in papers:
            if paper['paperId'] in new:
                await self.put_todo(paper)

    async def put_todo(self, paper: dict):
        # paper is a dict with fields like paper_id, title, abstract, etc.
        if self.total >= self.max_papers:
            return
        self.total += 1
        await self.todo.put(paper)


async def main():
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
            max_papers=1000,
        )
        await crawler.run()
    end = time.perf_counter()

    seen = sorted(crawler.seen)
    print("Results:")
    for paper in seen:
        print(paper)
    print(f"Crawled: {len(crawler.done)} Papers")
    print(f"Found: {len(seen)} Papers")
    print(f"Done in {end - start:.2f}s")


if __name__ == '__main__':
    asyncio.run(main(), debug=True)


async def homework():
    """
    Ideas for you to implement to test your understanding:
    - Respect robots.txt *IMPORTANT*
    - Find all links in sitemap.xml
    - Provide a user agent
    - Normalize urls (make sure not to count mcoding.io and mcoding.io/ as separate)
    - Skip filetypes (jpg, pdf, etc.) or include only filetypes (html, php, etc.)
    - Max depth
    - Max concurrent connections per domain
    - Rate limiting
    - Rate limiting per domain
    - Store connections as graph
    - Store results to database
    - Scale
    """
