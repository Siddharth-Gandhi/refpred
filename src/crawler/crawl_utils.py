'''
Contains various utility functions.
'''

def get_reference_url(paper_id, is_arxiv: bool = False) -> str:
        """Get the URL for the references of a paper"""
        if is_arxiv:
            return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}/references?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"
        return f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"

def get_paper_url(paper_id, is_arxiv: bool = False) -> str:
        """Get the URL for a paper"""
        if is_arxiv:
            return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"
        return f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles"


def get_batch_url() -> str:
    """Get the URL for a batch of papers"""
    return 'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,url,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles'
