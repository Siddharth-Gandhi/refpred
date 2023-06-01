

// function createCard(paper) {
//     const card = document.createElement('div');
//     card.className = 'card';

//     const title = document.createElement('h2');
//     title.textContent = paper.title;
//     card.appendChild(title);

//     // const authors = document.createElement('p');
//     // authors.className = 'authors';
//     // authors.textContent = `Authors: ${paper.authors}`;
//     // card.appendChild(authors);

//     const abstract = document.createElement('p');
//     // abstract.textContent = paper.abstract.slice(0, 500) + '...';
//     // abstract.textContent = paper.abstract ? paper.abstract.slice(0, 100) + '...' : 'Abstract: Not Available';
//     abstract.textContent = paper.abstract || 'Abstract: Not Available in Database';

//     card.appendChild(abstract);

//     const year = document.createElement('p');
//     year.className = 'year';
//     year.textContent = `Year: ${paper.year}`;
//     card.appendChild(year);

//     const score = document.createElement('p');
//     score.className = 'score';
//     score.textContent = `Score: ${paper.score * 100}`;
//     card.appendChild(score);

//     if (paper.url) {
//         // TODO This will always be null
//         const url = document.createElement('a');
//         url.className = 'url';
//         url.href = paper.url;
//         url.target = '_blank';
//         url.textContent = 'Read Paper';
//         card.appendChild(url);
//     }

//     return card;
// }


// function createCard(paper, index) {
//     const card = document.createElement('div');
//     card.className = 'card';
//     card.id = `card-${index}`;

//     const title = document.createElement('h2');
//     title.textContent = `${index}. ${paper.title}`;
//     card.appendChild(title);

//     const abstract = document.createElement('p');
//     abstract.className = 'abstract';
//     abstract.textContent = paper.abstract || 'Abstract: Not Available in Database';
//     card.appendChild(abstract);

//     const year = document.createElement('p');
//     year.className = 'year';
//     year.textContent = `Year: ${paper.year}`;
//     card.appendChild(year);

//     const score = document.createElement('p');
//     score.className = 'score';
//     const formattedScore = (paper.score * 100).toFixed(3).slice(0, -1);

//     score.textContent = `Score: ${formattedScore}`;
//     card.appendChild(score);

//     card.addEventListener('click', () => {
//         card.classList.toggle('clicked');
//         abstract.style.display = abstract.style.display === 'none' ? 'block' : 'none';
//         bibtexButton.style.display = abstract.style.display;
//     });




//     const bibtexButton = document.createElement('button');
//     bibtexButton.textContent = 'Bibtex';
//     bibtexButton.className = 'bibtex-button';
//     bibtexButton.style.display = 'none'; // initially hidden
//     card.appendChild(bibtexButton);


//     const bibtexBox = document.createElement('div');
//     bibtexBox.className = 'bibtex-box';
//     bibtexBox.style.display = 'none';
//     card.appendChild(bibtexBox);

//     bibtexButton.addEventListener('click', () => {
//         event.stopPropagation();
//         if (paper.citationStyles && paper.citationStyles.bibtex) {
//             bibtexBox.textContent = paper.citationStyles.bibtex;
//         } else {
//             bibtexBox.textContent = 'BibTex not available';
//         }
//         bibtexBox.style.display = bibtexBox.style.display === 'none' ? 'block' : 'none';
//     });



//     return card;
// }


// function createCard(paper, index) {
//     console.log(paper);
//     const card = document.createElement('div');
//     card.className = 'card';
//     card.id = `card-${index}`;

//     const title = document.createElement('h2');
//     title.textContent = `${index}. ${paper.title}`;
//     card.appendChild(title);

//     const abstract = document.createElement('p');
//     abstract.className = 'abstract';
//     abstract.textContent = paper.abstract || 'Abstract: Not Available in Database';
//     card.appendChild(abstract);

//     const year = document.createElement('p');
//     year.className = 'year';
//     year.textContent = `Year: ${paper.year}`;
//     card.appendChild(year);

//     const score = document.createElement('p');
//     score.className = 'score';
//     const formattedScore = (paper.score * 100).toFixed(3).slice(0, -1);

//     score.textContent = `Score: ${formattedScore}`;
//     card.appendChild(score);

//     const bibtexButton = document.createElement('button');
//     bibtexButton.textContent = 'Bibtex';
//     bibtexButton.className = 'bibtex-button';
//     bibtexButton.style.display = 'none';
//     card.appendChild(bibtexButton);

//     const bibtexBox = document.createElement('div');
//     bibtexBox.className = 'bibtex-box';
//     bibtexBox.style.display = 'none';
//     card.appendChild(bibtexBox);

//     const arrow = document.createElement('div');
//     arrow.className = 'expand-arrow';
//     arrow.innerHTML = '&#9660;'; // Down arrow
//     card.appendChild(arrow);


//     // card.addEventListener('click', (event) => {
//     //     if (event.target !== card) {
//     //         return;
//     //     }
//     //     card.classList.toggle('clicked');
//     //     if (abstract.style.display === 'none' || abstract.style.display === '') {
//     //         abstract.style.display = 'block';
//     //         bibtexButton.style.display = 'block';
//     //     } else {
//     //         abstract.style.display = 'none';
//     //         bibtexButton.style.display = 'none';
//     //         bibtexBox.style.display = 'none';
//     //     }
//     // });

//     card.addEventListener('click', (event) => {
//     // Check if the clicked element is a child of the card
//     const isChild = card.contains(event.target);

//     if (!isChild || event.target === bibtexButton) {
//         return;
//     }

//     card.classList.toggle('clicked');
//     if (abstract.style.display === 'none' || abstract.style.display === '') {
//         abstract.style.display = 'block';
//         bibtexButton.style.display = 'block';
//     } else {
//         abstract.style.display = 'none';
//         bibtexButton.style.display = 'none';
//         bibtexBox.style.display = 'none';
//     }
// });


//     bibtexButton.addEventListener('click', () => {
//         if (paper.citationStyles && paper.citationStyles.bibtex) {
//             bibtexBox.textContent = paper.citationStyles.bibtex;
//         } else {
//             bibtexBox.textContent = 'BibTex not available';
//         }
//         bibtexBox.style.display = bibtexBox.style.display === 'none' ? 'block' : 'none';
//     });

//     arrow.addEventListener('click', () => {
//         card.classList.toggle('clicked');
//         if (abstract.style.display === 'none' || abstract.style.display === '') {
//             abstract.style.display = 'block';
//             bibtexButton.style.display = 'block';
//             arrow.innerHTML = '&#9650;'; // Up arrow
//         } else {
//             abstract.style.display = 'none';
//             bibtexButton.style.display = 'none';
//             bibtexBox.style.display = 'none';
//             arrow.innerHTML = '&#9660;'; // Down arrow
//         }
//         });


//     return card;
// }

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

function createCard(paper, index) {
    console.log(paper);
    const card = document.createElement('div');
    card.className = 'card';
    card.id = `card-${index}`;
    let isExpanded = false

    const title = document.createElement('h2');
    title.textContent = `${index}. ${paper.title}`;
    card.appendChild(title);

    const abstract = document.createElement('p');
    abstract.className = 'abstract';
    abstract.textContent = paper.abstract || 'Abstract: Not Available in Database';
    card.appendChild(abstract);

    const year = document.createElement('p');
    year.className = 'year';
    year.textContent = `Year: ${paper.year}`;
    card.appendChild(year);

    const score = document.createElement('p');
    score.className = 'score';
    const formattedScore = (paper.score * 100).toFixed(3).slice(0, -1);

    score.textContent = `Score: ${formattedScore}`;
    card.appendChild(score);

    const bibtexButton = document.createElement('button');
    bibtexButton.textContent = 'Bibtex';
    bibtexButton.className = 'bibtex-button';
    bibtexButton.style.display = 'none';
    card.appendChild(bibtexButton);

    const bibtexBox = document.createElement('div');
    bibtexBox.className = 'bibtex-box';
    bibtexBox.style.display = 'none';
    card.appendChild(bibtexBox);

    const arrow = document.createElement('div');
    arrow.className = 'expand-arrow';
    arrow.innerHTML = '&#9660;'; // Down arrow
    card.appendChild(arrow);

    // card.addEventListener('click', (event) => {
    //     const isChild = card.contains(event.target);

    //     if (!isChild || event.target === bibtexButton || event.target === arrow) {
    //         return;
    //     }

    //     card.classList.toggle('clicked');
    //     if (abstract.style.display === 'none' || abstract.style.display === '') {
    //         abstract.style.display = 'block';
    //         bibtexButton.style.display = 'block';
    //     } else {
    //         abstract.style.display = 'none';
    //         bibtexButton.style.display = 'none';
    //         bibtexBox.style.display = 'none';
    //     }
    // });

    arrow.addEventListener('click', (event) => {
        event.stopPropagation();
        card.classList.toggle('clicked');
        isExpanded = !isExpanded;
        if (isExpanded) {
            abstract.style.display = 'block';
            bibtexButton.style.display = 'block';
            arrow.innerHTML = '&#9650;'; // Up arrow
        } else {
            abstract.style.display = 'none';
            bibtexButton.style.display = 'none';
            bibtexBox.style.display = 'none';
            arrow.innerHTML = '&#9660;'; // Down arrow
        }
    });


    bibtexButton.addEventListener('click', () => {
        if (paper.citationStyles && paper.citationStyles.bibtex) {
            bibtexBox.textContent = paper.citationStyles.bibtex;
        } else {
            bibtexBox.textContent = 'BibTex not available';
        }
        bibtexBox.style.display = bibtexBox.style.display === 'none' ? 'block' : 'none';
    });

    return card;
}




function displayResults(recommendations) {
    const results = document.getElementById('results');
    results.innerHTML = '';

    // recommendations.forEach((paper) => {
    //     const card = createCard(paper);
    //     results.appendChild(card);
    // });
    recommendations.forEach((paper, index) => {
    const card = createCard(paper, index + 1);
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
