from dash.dependencies import Input, Output, State
from webapp import app, logger
from webapp.figure_maker import (
    make_figure, prepare_materials, get_bmu,
    PAPER_COLOR, WORD_COLOR,
)
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



import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


def make_paper_component(title, abst, url, rank):
    return html.Div([
        rank,
        html.A(title, href=url, target='blank'),
        html.P(abst)
    ])


@app.callback([
        Output('paper-list', 'children'),
        Output('paper-list', 'style'),
    ],
    [
        Input('paper-map', 'clickData'),
        Input('word-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('paper-list', 'style'),
    ],
    prevent_initial_call=True
)
def make_paper_list(paperClickData, wordClickData, keyword, style):
    logger.debug('make_paper_list')

    ctx = dash.callback_context
    map_name = ctx.triggered[0]['prop_id'].split('.')[0]
    logger.info(f"map_name: {map_name}")

    df, labels, _, history, _ = prepare_materials(keyword, 'TSOM')
    paper_labels = labels[0].values.tolist()
    if map_name == 'paper-map':
        clicked_point = [[paperClickData['points'][0]['x'], paperClickData['points'][0]['y']]] if paperClickData else [[0, 0]]
        clicked_point = np.array(clicked_point)
        dists = dist.cdist(history['Z1'], clicked_point)
        paper_idxs = np.argsort(dists, axis=0)[:3].flatten()
    else:
        clicked_point = [[wordClickData['points'][0]['x'], wordClickData['points'][0]['y']]] if wordClickData else [[0, 0]]
        clicked_point = np.array(clicked_point)
        logger.debug(clicked_point)
        y = history['Y'][:, get_bmu(history['Zeta'], wordClickData)]
        target_nodes = (-y).flatten().argsort()[:3]
        logger.debug(f"target_nodes: {target_nodes}")
        paper_idxs = []
        for k in target_nodes:
            _dists = dist.cdist(history['Z1'], history['Zeta'][k, None])
            _paper_idxs = np.argsort(_dists, axis=0)[:3].flatten().tolist()
            paper_idxs.extend(_paper_idxs)
        seen = set()
        seen_add = seen.add
        paper_idxs = [idx for idx in paper_idxs if not (idx in seen or seen_add(idx))]
    logger.debug(f"Paper indexes {paper_idxs}")
    layout = [
        make_paper_component(paper_labels[i], df['snippet'][i], df['URL'][i], df['ranking'][i]) for i in paper_idxs
    ]
    style['borderColor'] = PAPER_COLOR if map_name == 'paper-map' else WORD_COLOR

    return layout, style
