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
        bibtexButton.textContent = 'BibTeX';
        bibtexButton.className = 'bibtex-button';
        bibtexButton.style.display = 'none';
        card.appendChild(bibtexButton);

        const bibtexBox = document.createElement('div');
        bibtexBox.className = 'bibtex-box';
        bibtexBox.style.display = 'none';
        card.appendChild(bibtexBox);

        const expandButton = document.createElement('button');
        expandButton.textContent = 'EXPAND';
        expandButton.className = 'expand-button';
        card.appendChild(expandButton);

        expandButton.addEventListener('click', (event) => {
            event.stopPropagation();
            card.classList.toggle('clicked');
            isExpanded = !isExpanded;
            if (isExpanded) {
                abstract.style.display = 'block';
                bibtexButton.style.display = 'block';
                expandButton.textContent = 'MINIMIZE'; // Change text to 'Minimize'
            } else {
                abstract.style.display = 'none';
                bibtexButton.style.display = 'none';
                bibtexBox.style.display = 'none';
                expandButton.textContent = 'EXPAND'; // Change text back to 'Expand'
            }
        });

        // const arrow = document.createElement('div');
        // arrow.className = 'expand-arrow';
        // arrow.innerHTML = '&#9660;'; // Down arrow
        // card.appendChild(arrow);

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

        bibtexButton.addEventListener('click', () => {
            if (paper.citationStyles && paper.citationStyles.bibtex) {
                bibtexBox.textContent = paper.citationStyles.bibtex;
            } else {
                bibtexBox.textContent = 'BibTex not available';
            }
            bibtexBox.style.display = bibtexBox.style.display === 'none' ? 'block' : 'none';
        });

        const infoWrapper = document.createElement('div');
        infoWrapper.className = 'info-wrapper';
        infoWrapper.style.alignItems = 'center';

        infoWrapper.appendChild(abstract);
        infoWrapper.appendChild(year);
        infoWrapper.appendChild(score);
        infoWrapper.appendChild(bibtexBox);

        const buttonWrapper = document.createElement('div');

        buttonWrapper.className = 'button-wrapper';
        buttonWrapper.style.display = 'flex';
        buttonWrapper.style.justifyContent = 'space-between';
        buttonWrapper.appendChild(expandButton);
        buttonWrapper.appendChild(bibtexButton);


        infoWrapper.appendChild(buttonWrapper);
        card.appendChild(infoWrapper);

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
