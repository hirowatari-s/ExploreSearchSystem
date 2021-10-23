from dash.dependencies import Input, Output, State
from webapp import app, logger
import pandas as pd
from webapp.figure_maker import make_figure
from webapp.dash import main_layout, landing_page_layout

from functools import partial


def load_learning(viewer_id, n_clicks, viewer_name, clickData, keyword, prev_fig):
    logger.debug(f"graph '{viewer_name}' clicked")
    logger.debug(f"clickData: {clickData}")
    keyword = keyword or "Machine Learning"
    if clickData and "points" in clickData and "pointIndex" in clickData["points"][0]:
        viewer_name = "CCP"
    return make_figure(keyword, viewer_name, viewer_id, clickData)


app.callback(
    Output('paper-map', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('viewer-selector', 'value'),
        Input('word-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('paper-map', 'figure'),
])(partial(load_learning, "viewer_1"))

app.callback(
    Output('word-map', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('viewer-selector', 'value'),
        Input('paper-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('word-map', 'figure'),
])(partial(load_learning, "viewer_2"))


@app.callback([
        Output('main', 'style'),
        Output('landing', 'style'),
        Output('paper-map-col', 'style'),
        Output('word-map-col', 'style'),
        Output('search-form', 'value'),
        Output('viewer-selector', 'value'),
    ], [
        Input('landing-explore-start', 'n_clicks'),
    ], [
        State('landing-search-form', 'value'),
    ], prevent_initial_call=True)
def make_page(n_clicks, keyword):
    logger.info(f"first search started with keyword: {keyword}")
    main_style = {}
    landing_style = {}
    paper_style = {"height": "100%"}
    word_style = {"height": "100%"}

    main_style['display'] = 'block'
    landing_style['display'] = 'none'
    paper_style['display'] = 'block'
    word_style['display'] = 'block'

    keyword = keyword or 'Machine Learning'

    return main_style, landing_style, paper_style, word_style, keyword, 'CCP'

