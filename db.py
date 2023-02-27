from dataclasses import dataclass, field

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


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
