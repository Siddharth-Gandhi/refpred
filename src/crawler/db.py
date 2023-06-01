"""
Handles the MongoDB database and it's operations.
"""

import logging
import logging.config
from dataclasses import dataclass, field
from typing import Set

import pymongo
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
    init_new: bool = field(default=False)
    client: MongoClient = field(init=False)
    db: Database = field(init=False)
    collection: Collection = field(init=False)
    stored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize the MongoDB database and collection"""
        # self.client = MongoClient(self.mongo_url)
        object.__setattr__(self, "client", MongoClient(self.mongo_url))
        # self.db = self.client[self.db_name]
        object.__setattr__(self, "db", self.client[self.db_name])
        all_collections = self.db.list_collection_names()
        if self.init_new and self.collection_name in all_collections:
            logger.warning(f"Dropped pre-existing '{self.collection_name}' collection")
            self.db.drop_collection(self.collection_name)
            logger.info(f"Creating '{self.collection_name}' collection")
        # self.collection = self.db[self.collection_name]
        object.__setattr__(self, "collection", self.db[self.collection_name])

    def insert_one(self, document: dict) -> None:
        """Insert a document into the collection"""
        self.collection.insert_one(document)
        # self.stored += 1
        object.__setattr__(self, "stored", self.stored + 1)
        if self.stored % 100 == 0:
            logger.info(f"Inserted {self.stored} documents")

    # def insert_many(self, documents: list) -> None:
    #     """Insert multiple documents into the collection"""
    #     self.collection.insert_many(documents)
    #     # self.stored += len(documents)
    #     object.__setattr__(self, "stored", self.stored + len(documents))
    #     if self.stored % 100 == 0:
    #         logger.info(f"Inserted {self.stored} documents")

    def insert_many(self, documents: list) -> None:
        """Insert multiple documents into the collection or update if _id already exists"""

        ##### DEBUG #####
        # for doc in documents:
        #     print(f"ID to insert/update: {doc['_id']}")

        # doc_id_to_check = documents[0]["_id"]
        # doc_in_db = self.collection.find_one({"_id": doc_id_to_check})
        # print(f"Doc in DB with ID {doc_id_to_check}: {doc_in_db}")
        ##### /DEBUG #####

        bulk_operations = [
            pymongo.UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True) for doc in documents
        ]
        result = self.collection.bulk_write(bulk_operations)
        ##### DEBUG #####
        # try:
        #     result = self.collection.bulk_write(bulk_operations)
        # except pymongo.errors.BulkWriteError as bwe:
        #     print(bwe.details)
        ##### /DEBUG #####

        print(result.upserted_count, result.modified_count, result.acknowledged)
        upserted_count = result.upserted_count
        modified_count = result.modified_count

        object.__setattr__(self, "stored", self.stored + upserted_count + modified_count)

        if self.stored % 100 == 0:
            logger.info(
                f"Inserted {upserted_count} new documents and modified {modified_count} existing documents. Total documents: {self.stored}"
            )

    def get_ids(self) -> Set[str]:
        """Get all the ids in the collection"""
        ids = self.collection.find({}, {"_id": 1})
        return {obj["_id"] for obj in ids}
