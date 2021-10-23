from dash.dependencies import Input, Output, State
from webapp import app, logger
from webapp.figure_maker import make_figure, prepare_materials
from functools import partial

import numpy as np
from scipy.spatial import distance as dist


def load_learning(viewer_id, n_clicks, viewer_name, clickData, keyword, prev_fig):
    logger.debug(f"graph '{viewer_name}' clicked")
    logger.debug(f"clickData: {clickData}")
    keyword = keyword or "Machine Learning"
    if clickData and "points" in clickData and "pointIndex" in clickData["points"][0]:
        viewer_name = "CCP"
    return make_figure(keyword, viewer_name, viewer_id, clickData)


app.callback(Output('paper-map', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('viewer-selector', 'value'),
        Input('word-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('paper-map', 'figure'),
])(partial(load_learning, "viewer_1"))

app.callback(Output('word-map', 'figure'),
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



import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

@app.callback(
    Output('paper-list', 'children'),
    [
        Input('paper-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
    ],
    prevent_initial_call=True
)
def make_paper_list(clickData, keyword):
    logger.debug('make_paper_list')
    _, labels, _, history, _ = prepare_materials(keyword, 'TSOM')
    clicked_point = [[clickData['points'][0]['x'], clickData['points'][0]['y']]] if clickData else [[0, 0]]
    clicked_point = np.array(clicked_point)
    dists = dist.cdist(history['Z1'], clicked_point)
    paper_idxs = np.argsort(dists, axis=0)[:3].flatten()
    paper_labels = labels[0].values.tolist()
    layout = [f"{i}:{paper_labels[i]}" for i in paper_idxs]
    print(layout)

    return layout
