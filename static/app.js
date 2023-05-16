async function fetchRecommendations(title, abstract, num_papers) {
    const response = await fetch('http://127.0.0.1:5000/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, abstract, num_papers}),
    });

    return await response.json();
}

function createCard(paper) {
    const card = document.createElement('div');
    card.className = 'card';

    const title = document.createElement('h2');
    title.textContent = paper.title;
    card.appendChild(title);

    // const authors = document.createElement('p');
    // authors.className = 'authors';
    // authors.textContent = `Authors: ${paper.authors}`;
    // card.appendChild(authors);

    const abstract = document.createElement('p');
    // abstract.textContent = paper.abstract.slice(0, 500) + '...';
    // abstract.textContent = paper.abstract ? paper.abstract.slice(0, 500) + '...' : 'Abstract: Not Available';
    abstract.textContent = paper.abstract || 'Abstract: Not Available in Database';


    card.appendChild(abstract);

    const year = document.createElement('p');
    year.className = 'year';
    year.textContent = `Year: ${paper.year}`;
    card.appendChild(year);

    const score = document.createElement('p');
    score.className = 'score';
    score.textContent = `Score: ${paper.score * 100}`;
    card.appendChild(score);

    if (paper.url) {
        // TODO This will always be null
        const url = document.createElement('a');
        url.className = 'url';
        url.href = paper.url;
        url.target = '_blank';
        url.textContent = 'Read Paper';
        card.appendChild(url);
    }

    return card;
}



function displayResults(recommendations) {
    const results = document.getElementById('results');
    results.innerHTML = '';

    recommendations.forEach((paper) => {
        const card = createCard(paper);
        results.appendChild(card);
    });
}

document.getElementById('search-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const title = document.getElementById('title').value;
    const abstract = document.getElementById('abstract').value;
    const num_papers = document.getElementById('num_papers').value;

    const recommendations = await fetchRecommendations(title, abstract, num_papers);
    console.log(recommendations);
    displayResults(recommendations);
});
