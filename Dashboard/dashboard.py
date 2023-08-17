
from dash import Dash, html, dcc, Input, Output, callback, callback_context, no_update
import dash_bootstrap_components as dbc
import dash_daq as daq
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

#Load the dataset that we can use locally
train_data = pd.read_csv("./sampled_train_data.csv", index_col=0)
col_description = pd.read_csv('./HomeCredit_columns_description.csv', encoding='ISO-8859-1', index_col=0)
train_data['TARGET'] = 1-train_data['TARGET'] #Make it so 1 is the positive class

# region API Calls
api_ipv4 = '16.171.250.228'
def get_global_importance():
    response = requests.get(f'http://{api_ipv4}/global_importance')
    if response.status_code == 200:
        with open("Dashboard/assets/shap_summary.png", "wb") as f:
            f.write(response.content)

def predict_proba(X1):
    data = X1.to_json(orient='records')
    headers = {'Content-type': 'application/json'}
    response = requests.post(f'http://{api_ipv4}/predict', data=data, headers=headers)
    return response.json()

def get_force_plot(X1):
    data = X1.to_json(orient='records')
    headers = {'Content-type': 'application/json'}
    response = requests.post(f'http://{api_ipv4}/get_local_importance', data=data, headers=headers)
    if response.status_code == 200:
        with open("Dashboard/assets/local_importance.html", "wb") as f:
            f.write(response.content)

def get_threshold():
    response = requests.get(f'http://{api_ipv4}/threshold')
    if response.status_code == 200:
        return response.json()
#endregion
#region helper function
def prepare_data_for_model(df, dataset_version:str):

    if dataset_version == '2.0':
        df.columns = df.columns.str.replace('[^\w\s]','')
        not_usable_col = ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']
        df.drop(columns='level_0', inplace=True)
        columns_X = [col for col in df.columns if col not in not_usable_col]
        df = df[columns_X]
    return df
#endregion

#region Components:
app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
server = app.server

title = dcc.Markdown(children='# Prêt à dépenser, octroi de crédit')

client_id = dcc.Dropdown(
                 id='client-id-dropdown',
                 options=[{'label': str(id), 'value': id} for id in train_data['SK_ID_CURR'].unique()],
                 value=train_data['SK_ID_CURR'].iloc[0],
                 clearable=False,
                 persistence=True,
                 persistence_type='session'
                 ) # The client id maybe a drop down with all the values avaialble in train_data['SK_ID_CURR']
 
gauge = daq.Thermometer(id='thermometer-gauge',
                    value=50,
                    max=100,
                    scale={'start': 0, 'interval': 20,
                    'labelInterval': 1, 'custom': {'50': 'Goal'}},
                    showCurrentValue=True) #Dash DAQ > Thermometer fill it base on the prediction% got from the API call to know how far the client is from the treshold

client_force_plot = html.Iframe(id='local-importance-frame', srcDoc='', width='100%', height='400') # The force plot, got from the api, this is a html file

global_feature_importance_plot = html.Img(id='feature-importance-img', src='shap_summary.png') # the globale feature importance of the model get from the api this is a png
feature_options = [{"label": col, "value": col} for col in train_data.columns]

# Rest of the data will be from the two dataframe available here
feature_dropdown1 = dcc.Dropdown(
                        id="feature_dropdown1",
                        options=feature_options,
                        value=feature_options[7]['value'],  # Default value set to the first column
                        persistence=True,
                        persistence_type='session'
                    ) # Select a feature from all available column in the df
feature_description_1 = html.Div(id="feature_description_1", children='') # the description of the selected feature taken from the description dataframe
feature_dropdown2 = dcc.Dropdown(
                        id="feature_dropdown2",
                        options=feature_options,
                        value=feature_options[10]['value'],  # Default value set to the first column
                        persistence=True,
                        persistence_type='session'
                    ) # select a second feature from all available column in the df
feature_description_2 = html.Div(id="feature_description_2", children='')# the description of the selected feature taken from the description dataframe
graph_feature1 = dcc.Graph(id="graph_feature1", figure={})# distribution of the feature1 based on class - Put the client row/data on evidence
graph_feature2 = dcc.Graph(id="graph_feature2", figure={})# distribution of the feature2 based on class - Put the client row/data on evidence
graph_bivariate = dcc.Graph(id="graph_bivariate", figure={})# bivariate plot of the two class - Put the client on evidence
#endregion

#region Layout:
# App Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(title, width={"size": 6, "offset": 3}, className="text-center")  # Center the title in a column of size 6 offset by 3.
    ], className="mb-4"),  # Add some margin-bottom for spacing

    dbc.Row([
        dbc.Col([
            html.Div(client_id, style={'marginBottom': '50px'}),  # Adding margin to the bottom of client_id
            client_force_plot], width=10),  # Stack client_id and client_force_plot in a half-width column
        dbc.Col(gauge, width=2)  # Half-width column for gauge
    ], className="mb-4"),  # Add some margin-bottom for spacing

        dbc.Row([
        dbc.Col(width=12, children=[
            dbc.Tabs([
                dbc.Tab(label="Global feature importance", children=[
                    global_feature_importance_plot
                ]),
                dbc.Tab(label="Feature Analysis", children=[
                    feature_dropdown1,
                    feature_description_1,
                    feature_dropdown2,
                    feature_description_2,
                    graph_feature1,
                    graph_feature2
                ]),
                dbc.Tab(label="Bivariate Analysis", children=[
                    graph_bivariate
                ]),
            ])
        ])
    ])
])
#endregion


#Callbacks update_client_id
@app.callback(
    [Output('thermometer-gauge', 'value'),
    Output('thermometer-gauge', 'scale'),
    Output('feature-importance-img', 'src'),
    Output('local-importance-frame', 'srcDoc')],
    Input('client-id-dropdown', 'value'),
    
)
def update_client_id(client_id):
    ctx = callback_context
    if not ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f"Triggered by: {trigger_id}")
        return no_update
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Triggered by: {trigger_id}")


    X1 = train_data[train_data['SK_ID_CURR'] == client_id]
    X1 = prepare_data_for_model(X1, '2.0')
    
    # Get Global Importance
    get_global_importance()

    # Predict Proba
    predictions = predict_proba(X1)
    class1_proba = predictions[0][2] * 100

    # Get threshold 
    threshold = get_threshold()
    goal = (1 - float(threshold))*100
    scale={'start': 0, 'interval': 20,
                    'labelInterval': 1, 'custom': {str(goal): 'Goal'}}

    # Get Force Plot
    get_force_plot(X1)
    with open('Dashboard/assets/local_importance.html', 'r') as file:
        force_plot_content = file.read()

    return class1_proba, scale, '/assets/shap_summary.png', force_plot_content


#Callbacks update_feature1_and_graph
@app.callback(
    [Output('feature_description_1', 'children'),
     Output('graph_feature1', 'figure')],
    [Input('feature_dropdown1', 'value'),
     Input('client-id-dropdown', 'value')] 
)
def update_feature1_and_graph(selected_feature, client_id):
    # Description
    try:
        description = col_description[col_description['Row'] == selected_feature]['Description'].iloc[0]
    except IndexError:
        description = "Description not found for this feature."

    # Graph
    fig = px.histogram(train_data, x=selected_feature, color='TARGET', barmode='overlay', histnorm='percent')
    fig.update_layout(title=f"Distribution of {selected_feature} by TARGET", xaxis_title=selected_feature, yaxis_title="Percentage", barmode='group')
    client_feature_value = train_data[train_data['SK_ID_CURR'] == client_id][selected_feature].iloc[0]
    if pd.notna(client_feature_value):
        fig.add_shape(
            type="line",
            x0=client_feature_value,
            x1=client_feature_value,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="black", width=2),
            name="Client Value"
        )

    return description, fig


#Callbacks update_feature2_and_graph
@app.callback(
    [Output('feature_description_2', 'children'),
     Output('graph_feature2', 'figure')],
    [Input('feature_dropdown2', 'value'),
     Input('client-id-dropdown', 'value')] 
)
def update_feature2_and_graph(selected_feature, client_id):
    # Description
    try:
        description = col_description[col_description['Row'] == selected_feature]['Description'].iloc[0]
    except IndexError:
        description = "Description not found for this feature."

    # Graph
    fig = px.histogram(train_data, x=selected_feature, color='TARGET', barmode='overlay', histnorm='percent')
    fig.update_layout(title=f"Distribution of {selected_feature} by TARGET", xaxis_title=selected_feature, yaxis_title="Percentage", barmode='group')
    # Extracting the feature value for the selected client
    client_feature_value = train_data[train_data['SK_ID_CURR'] == client_id][selected_feature].iloc[0]
    if pd.notna(client_feature_value):
        fig.add_shape(
            type="line",
            x0=client_feature_value,
            x1=client_feature_value,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="black", width=2),
            name="Client Value"
        )
    return description, fig



#Callbacks update_bivariate_plot
@app.callback(
    Output('graph_bivariate', 'figure'),
    [Input('feature_dropdown1', 'value'),
     Input('feature_dropdown2', 'value'),
     Input('client-id-dropdown', 'value')]
)
def update_bivariate_plot(feature1, feature2, client_id):

    # 1. Extract the selected features
    subset = train_data[[feature1, feature2, 'TARGET']]
    
    # 2. Plot these using a scatter plot with Plotly Express
    fig = px.scatter(subset, x=feature1, y=feature2, color='TARGET',
                     color_continuous_scale=["red", "blue"], 
                     #labels={'color': 'TARGET'},
                     title=f"{feature1} vs {feature2}")

    # 3. Highlight the particular client
    client_row = train_data[train_data['SK_ID_CURR'] == client_id]
    if pd.notna(client_row[feature1].iloc[0]) and pd.notna(client_row[feature2].iloc[0]):
        fig.add_trace(
            go.Scatter(x=client_row[feature1], y=client_row[feature2],
                       mode='markers',
                       marker=dict(color="yellow", size=20),
                       name=f"Client {client_id}")
                       )
    
    return fig



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)