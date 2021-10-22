const onLatentClicked = (_) => {
    const url = document.getElementById("link").href;
    if (url != "#")
        window.open(url);
}


const onFirstSearchSubmitted = () => {
    const mainStyle = document.getElementById('main').style;
    const landingStyle = document.getElementById('landing').style;
    const paperMapStyle = document.getElementById('paper-map-col').style;
    const wordMapStyle = document.getElementById('word-map-col').style;
    const landingSearchKeyword = document.getElementById('landing-search-form').value;

    mainStyle.display = 'block';
    landingStyle.display = 'none';
    paperMapStyle.display = 'block';
    wordMapStyle.display = 'block';
    document.getElementById('search-form').value = landingSearchKeyword;
    // document.getElementById('explore-start').click();

    return mainStyle;
}
