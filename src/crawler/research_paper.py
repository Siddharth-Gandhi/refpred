'''A dataclass for a research paper object.'''

from dataclasses import dataclass, field
from typing import Dict, List

from db import MongoDBClient


@dataclass(frozen=True)
class ResearchPaper:
    '''A dataclass for a research paper object.'''
    # _id: str = field(init=False, default_factory=str)
    _id: str
    paperId: str
    externalIds: Dict[str, str | int]
    url: str
    title: str
    abstract: str
    venue: str
    publicationVenue: str
    year: int
    referenceCount: int
    citationCount: int
    influentialCitationCount: int
    isOpenAccess: bool
    openAccessPdf: Dict[str, str]
    fieldsOfStudy: Dict[int, str]
    s2FieldsOfStudy: Dict[int, Dict[str, str]]
    publicationTypes: Dict[int, str]
    publicationDate: str
    journal: Dict[str, str]
    citationStyles: Dict[str, str]
    authors: Dict[int, Dict[str, str]]
    references: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, '_id', self.paperId)

if __name__ == '__main__':
    mongo_client = MongoDBClient()
    paper = mongo_client.collection.find_one()
    print(paper)