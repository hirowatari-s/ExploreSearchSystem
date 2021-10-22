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
        Output('link', 'children'),
        Output('link', 'href'),
        Output('link', 'target'),
        Output('card-text', 'children'),
        Output('snippet-text', 'children'),
    ],
    [
        Input('paper-map', 'hoverData'),
    ],
    [
        State('search-form', 'value'),
        State('link', 'children'),
        State('link', 'href'),
        State('link', 'target'),
        State('card-text', 'children'),
        State('snippet-text', 'children')
    ])
def update_title(hoverData, keyword, prev_linktext, prev_url, prev_target, prev_page_title, prev_snippet):
    if hoverData:
        if not ("points" in hoverData and "pointIndex" in hoverData["points"][0]) \
            or keyword == None:
            link_title = prev_linktext
            url = prev_url
            target = prev_target
            page_title = prev_page_title
            snippet = prev_snippet
        else:
            csv_df = pd.read_csv(keyword+".csv")
            index = hoverData['points'][0]['pointIndex']
            link_title = "サイトへ Go"
            labels = csv_df['site_name']
            url = csv_df['URL'][index]
            target = "_blank"
            page_title = labels[index]
            snippet = csv_df['snippet'][index]
    else:
        link_title = "マウスを当ててみよう"
        url = "#"
        target = "_self"
        page_title = ""
        snippet = ""
    return link_title, url, target, page_title, snippet


# app.clientside_callback(
#     "onLatentClicked",
#     Output('explore-start', 'outline'),
#     Input('example-graph', 'clickData'), prevent_initial_call=True)


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def make_page(pathname):
    if pathname == '/':
        return landing_page_layout
    elif pathname == '/map':
        return main_layout
    else:
        return '404'
