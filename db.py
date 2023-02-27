"""
Handles the MongoDB database and it's operations.
"""

import logging
import logging.config
from dataclasses import dataclass, field
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

logging.config.fileConfig(fname="logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MongoDBClient:
    """A MongoDB database"""

    mongo_url: str = "mongodb://localhost:27017"
    db_name: str = "refpred"
    collection_name: str = "test"
    client: MongoClient = field(init=False)
    db: Database[Any] = field(init=False)
    collection: Collection[Any] = field(init=False)
    stored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize the MongoDB database and collection"""
        # self.client = MongoClient(self.mongo_url)
        object.__setattr__(self, "client", MongoClient(self.mongo_url))
        # self.db = self.client[self.db_name]
        object.__setattr__(self, "db", self.client[self.db_name])
        all_collections = self.db.list_collection_names()
        if self.collection_name in all_collections:
            logger.warning(f"Dropped pre-existing '{self.collection_name}' collection")
            self.db.drop_collection(self.collection_name)
        logger.info(f"Created '{self.collection_name}' collection")
        # self.collection = self.db[self.collection_name]
        object.__setattr__(self, "collection", self.db[self.collection_name])

    def insert_one(self, document: dict) -> None:
        """Insert a document into the collection"""
        self.collection.insert_one(document)
        # self.stored += 1
        object.__setattr__(self, "stored", self.stored + 1)
        if self.stored % 100 == 0:
            logger.info(f"Inserted {self.stored} documents")
        