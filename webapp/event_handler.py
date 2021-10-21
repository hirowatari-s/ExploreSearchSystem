from dash.dependencies import Input, Output, State
from webapp import app
import pandas as pd
from webapp.figure_maker import make_figure


@app.callback(
    Output('example-graph', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('model-selector', 'value'),
        Input('viewer-selector', 'value'),
        Input('example-graph2', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('example-graph', 'figure'),
    ])

def load_learning(n_clicks, model_name, viewer_name, clickData, keyword, prev_fig):
    print("clicked")
    print(clickData)
    print(keyword)

    if not keyword:
        keyword = "Machine Learning"

    if clickData and "points" in clickData and "pointIndex" in clickData["points"][0]:
        print("clicked_from_map2")
        # index = clickData['points'][0]['pointIndex']
        return make_figure(keyword, model_name, "CCP", "viewer_1", clickData)

    # return make_figure(keyword, model_name, "viewer_1", viewer_name)
    return make_figure(keyword, model_name, viewer_name, "viewer_1", None)


@app.callback(
    Output('example-graph2', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('model-selector', 'value'),
        Input('viewer-selector', 'value'),
        Input('example-graph', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('example-graph2', 'figure'),
    ])

def load_learning(n_clicks, model_name, viewer_name, clickData, keyword, prev_fig):
    if not keyword:
        keyword = "Machine Learning"

    if clickData:
        if not ("points" in clickData and "pointIndex" in clickData["points"][0]):
            pass
        else:
            print("clicked_from_map1")
            return make_figure(keyword, model_name, "CCP", "viewer_2", clickData)

    # return make_figure(keyword, model_name, "viewer_1", viewer_name)
    return make_figure(keyword, model_name, viewer_name, "viewer_2", None)


# def ccp_given_z2(clickData):
#         print("clicked")
#         # if not ("points" in Z1 and "pointIndex" in Z1["points"][0]) :
#         # if clicked viewer_2 variable, change viewer_1
#         return make_figure("ファッション", "TSOM", "viewer_1", viewer_name="CCP")



@app.callback([
        Output('link', 'children'),
        Output('link', 'href'),
        Output('link', 'target'),
        Output('card-text', 'children'),
        Output('snippet-text', 'children'),
    ],
    [
        Input('example-graph', 'hoverData'),
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


