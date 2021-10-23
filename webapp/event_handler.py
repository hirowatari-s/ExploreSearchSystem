import dash
from dash.dependencies import Input, Output, State
from webapp import app, logger
from webapp.figure_maker import make_figure
from functools import partial

@app.callback(
    Output('paper-map', 'figure'),
    Output('word-map', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('viewer-selector', 'value'),
        Input('paper-map', 'clickData'),
        Input('word-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('paper-map', 'figure'),
        State('word-map', 'figure'),
])

def load_learning(n_clicks, viewer_name, p_clickData, w_clickData, keyword, p_prev_fig, w_prev_fig):
    logger.debug(f"p_clickData: {p_clickData}")
    logger.debug(f"w_clickData: {w_clickData}")
    ctx = dash.callback_context
    print(ctx.triggered[0]['prop_id'])
    print(type(ctx.triggered[0]['prop_id']))

    viewer_1_name, viewer_2_name = viewer_name, viewer_name 
    if ctx.triggered[0]['prop_id'] == 'paper-map.clickData':
        if p_clickData and "points" in p_clickData and "pointIndex" in p_clickData["points"][0]:
            viewer_2_name = 'CCP' 
    elif ctx.triggered[0]['prop_id'] == 'word-map.clickData':
        if w_clickData and "points" in w_clickData and "pointIndex" in w_clickData["points"][0]:
            viewer_1_name = 'CCP' 

    keyword = keyword or "Machine Learning"
    return make_figure(keyword, viewer_1_name, 'viewer_1', w_clickData), make_figure(keyword, viewer_2_name, 'viewer_2', p_clickData)


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

