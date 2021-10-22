const onLatentClicked = (_) => {
    const url = document.getElementById("link").href;
    if (url != "#")
        window.open(url);
}


const onFirstSearchSubmitted = (e) => {
    const mainStyle = document.getElementById('main').style;
    const landingStyle = document.getElementById('landing').style;
    const paperMapStyle = document.getElementById('paper-map-col').style;
    const wordMapStyle = document.getElementById('word-map-col').style;
    const landingSearchKeyword = document.getElementById('landing-search-form').value;

    document.getElementById('search-form').value = landingSearchKeyword;
    landingStyle.display = 'none';
    paperMapStyle.display = 'block';
    wordMapStyle.display = 'block';
    mainStyle.display = 'block';
    const searchButton = document.getElementById('explore-start');
    console.dir(searchButton);
    // searchButton.onclick();

    return false;
}
