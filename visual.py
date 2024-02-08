import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tomotopy as tp
import plotly.graph_objs as go
from sklearn.datasets import fetch_20newsgroups
import dash_bootstrap_components as dbc
import dash_table
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


def calcular_coherencia(modelo_dtm, texts, num_topics, top_n, num_timepoints):
    texts = [text.split() for text in texts]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    coherence_values = []
    for k in range(num_topics):
        topic_words = []
        for t in range(num_timepoints):
            topic_words += [word for word, _ in modelo_dtm.get_topic_words(k, timepoint=t, top_n=top_n)]
        cm = CoherenceModel(topics=[topic_words], texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v')
        coherence_values.append(cm.get_coherence())
    return coherence_values

def extraer_topicos(modelo_dtm, timepoint, top_n):
    topicos = {}
    for k in range(modelo_dtm.k):
        palabras = modelo_dtm.get_topic_words(k, timepoint=timepoint, top_n=top_n)
        topicos[k] = [palabra for palabra, _ in palabras]
    return topicos


def visualizar_topicos_por_año(topicos_por_año):
    data = {}
    for año, topicos in topicos_por_año.items():
        if año not in data:
            data[año] = {}
        for num_topico, palabras in topicos.items():
            for indice, palabra in enumerate(palabras):
                if indice not in data[año]:
                    data[año][indice] = {}
                data[año][indice][f"Tópico {num_topico + 1}"] = palabra

    dataframes = []
    for año, topicos in data.items():
        df = pd.DataFrame(topicos).T
        df.index.name = 'Words'
        df.columns = [f"Tópico {i + 1}" for i in range(len(df.columns))]
        df.insert(0, 'Año', año)
        dataframes.append(df)

    result_df = pd.concat(dataframes)
    result_df = result_df.set_index(['Año', result_df.index])
    return result_df



def count_dominant_topics(modelo_dtm, num_topics, num_documentos_por_intervalo):
    dominant_topic_counts = np.zeros((len(num_documentos_por_intervalo), num_topics))
    
    start_idx = 0
    for interval_idx, num_docs in enumerate(num_documentos_por_intervalo):
        end_idx = start_idx + num_docs
        for doc_idx in range(start_idx, end_idx):
            topic_dist = modelo_dtm.docs[doc_idx].get_topic_dist()
            dominant_topic = np.argmax(topic_dist)
            dominant_topic_counts[interval_idx, dominant_topic] += 1
        start_idx = end_idx
    return dominant_topic_counts



def entrenar_modelo_dtm(datos, num_timepoints, num_topics, min_df, term_weight, total_iteraciones):
    modelo_dtm = tp.DTModel(tw=tp.TermWeight.IDF, k=num_topics, min_cf=min_df, t=num_timepoints)
    
    for index, documento in datos.iterrows():
        modelo_dtm.add_doc(documento['cleaned_text'].split(), timepoint=documento['timepoint'])
    for i in range(0, total_iteraciones, 10):
        modelo_dtm.train(10)
        print(f'Iteración: {i}, log-verosimilitud: {modelo_dtm.ll_per_word}')
        
    return modelo_dtm

def plot_topic_evolution_interac(modelo_dtm, num_topics, num_documentos_por_intervalo, intervalos_tiempo, topicos_por_año):
    num_timepoints = len(num_documentos_por_intervalo)
    timepoints = np.arange(num_timepoints)

    fig = go.Figure()

    for topic_idx in range(num_topics):
        topic_weights_over_time = []
        start_idx = 0
        for num_docs in num_documentos_por_intervalo:
            end_idx = start_idx + num_docs
            interval_weights = []
            for doc_idx in range(start_idx, end_idx):
                topic_dist = modelo_dtm.docs[doc_idx].get_topic_dist()
                interval_weights.append(topic_dist[topic_idx])
            topic_weights_over_time.append(np.mean(interval_weights))
            start_idx = end_idx

        top_keywords_over_time = [', '.join(topicos_por_año[year][topic_idx]) for year in intervalos_tiempo]

        fig.add_trace(go.Scatter(x=timepoints, y=topic_weights_over_time,
                    mode='lines+markers',
                    name=f'Tópico {topic_idx + 1}',
                    text=top_keywords_over_time,
                    hoverinfo='text+name'))

    fig.update_layout(
        title='Evolución de los tópicos a lo largo del tiempo',
        xaxis=dict(
            title='Años',
            tickvals=timepoints,
            ticktext=list(intervalos_tiempo),
        ),
        yaxis_title='Peso promedio del tópico'
    )

    return fig


def plot_dominant_topic_evolution_interac(dominant_topic_counts, intervalos_tiempo, topicos_por_año):
    num_timepoints, num_topics = dominant_topic_counts.shape
    timepoints = np.arange(num_timepoints)

    fig = go.Figure()

    for topic_idx in range(num_topics):
        top_keywords_over_time = [', '.join(topicos_por_año[year][topic_idx]) for year in intervalos_tiempo]

        fig.add_trace(go.Scatter(x=timepoints, y=dominant_topic_counts[:, topic_idx],
                    mode='lines+markers',
                    name=f'Tópico {topic_idx + 1}',
                    text=top_keywords_over_time,
                    hoverinfo='text+name'))

    fig.update_layout(
        title='Evolución de la cantidad de documentos dominantes por tópico a lo largo del tiempo',
        xaxis=dict(
            title='Años',
            tickvals=timepoints,
            ticktext=list(intervalos_tiempo),
        ),
        yaxis_title='Cantidad de documentos dominantes'
    )

    return fig

def plot_dominant_topic_evolution_stacked_interac(dominant_topic_counts, intervalos_tiempo, topicos_por_año):
    num_timepoints, num_topics = dominant_topic_counts.shape
    timepoints = np.arange(num_timepoints)

    fig = go.Figure()

    for topic_idx in range(num_topics):
        top_keywords_over_time = [', '.join(topicos_por_año[year][topic_idx]) for year in intervalos_tiempo]

        fig.add_trace(go.Scatter(x=timepoints, y=dominant_topic_counts[:, topic_idx],
                                 mode='lines',
                                 stackgroup='one',
                                 name=f'Tópico {topic_idx + 1}',
                                 text=top_keywords_over_time,
                                 hoverinfo='text+name'))

    fig.update_layout(
        title='Evolución de la cantidad de documentos dominantes por tópico a lo largo del tiempo (apilado)',
        xaxis=dict(
            title='Años',
            tickvals=timepoints,
            ticktext=list(intervalos_tiempo),
        ),
        yaxis_title='Cantidad de documentos dominantes'
    )

    return fig


def plot_stacked_bar_interac(modelo_dtm, num_topics, num_documentos_por_intervalo, intervalos_tiempo, topicos_por_año):
    num_timepoints = len(num_documentos_por_intervalo)
    timepoints = np.arange(num_timepoints)
    fig = go.Figure()

    for topic_idx in range(num_topics):
        topic_weights_over_time = []
        start_idx = 0
        for num_docs in num_documentos_por_intervalo:
            end_idx = start_idx + num_docs
            interval_weights = []
            for doc_idx in range(start_idx, end_idx):
                topic_dist = modelo_dtm.docs[doc_idx].get_topic_dist()
                interval_weights.append(topic_dist[topic_idx])
            topic_weights_over_time.append(np.mean(interval_weights))
            start_idx = end_idx

        top_keywords_over_time = [', '.join(topicos_por_año[year][topic_idx]) for year in intervalos_tiempo]

        fig.add_trace(go.Bar(x=timepoints, y=topic_weights_over_time, name=f'Tópico {topic_idx + 1}',
                    hovertext=top_keywords_over_time))


    fig.update_layout(
        barmode='stack',
        title='Evolución del peso de los tópicos a lo largo del tiempo',
        xaxis=dict(
            title='Años',
            tickvals=timepoints,
            ticktext=list(intervalos_tiempo),
        ),
        yaxis_title='Peso promedio del tópico'
    )

    return fig

def plot_coherence(coherences, num_topics):
    topic_labels = ['Tópico ' + str(i+1) for i in range(num_topics)]

    fig = go.Figure(data=[go.Bar(name='Coherencia', x=topic_labels, y=coherences)])

    fig.update_layout(
        title='Coherencia para ' + str(num_topics) + ' tópicos',
        xaxis_title='Tópico',
        yaxis_title='Coherencia'
    )

    return fig



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1('Modelado de tópicos temporales ', className='mb-4 text-center'),

    dbc.Row([
        dbc.Col([

            dcc.Upload(
                id='upload_data',
                children=html.Div([
                    html.A('selecciona archivos')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0px'
                },
                multiple=False
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Parámetros", style={'textAlign': 'center'}),
                    dbc.CardBody(
                        dbc.Form(
                            [
                                dbc.Row([
                                    dbc.Col(dbc.Label('Número de tópicos', html_for='num_topics'), width=6),
                                    dbc.Col(dbc.Input(id='num_topics', type='number', value=6, min=1), width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col(dbc.Label('Frecuencia mínima de aparición', html_for='min_df'), width=6),
                                    dbc.Col(dbc.Input(id='min_df', type='number', value=10, min=1), width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col(dbc.Label('Ponderación de términos', html_for='term_weight'), width=6),
                                    dbc.Col(dcc.Dropdown(
                                        id='term_weight',
                                        options=[{'label': 'IDF', 'value': 'IDF'},
                                                {'label': 'ONE', 'value': 'ONE'}],
                                        value='IDF'
                                    ), width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col(dbc.Label('Total de iteraciones para entrenar', html_for='total_iteraciones'), width=6),
                                    dbc.Col(dbc.Input(id='total_iteraciones', type='number', value=200, min=10), width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col(dbc.Label('Número de palabras claves', html_for='num_top_palabras'), width=6),
                                    dbc.Col(dbc.Input(id='num_top_palabras', type='number', value=5, min=1), width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col(dbc.Label("Calcular coherencia", html_for='toggle_coherence', className='mb-4'), width=6),
                                    dbc.Col(dbc.Checkbox(id='toggle_coherence', value=False, className='mb-2'), width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Button('Actualizar', id='update_button', color='primary'),
                                        className="d-flex justify-content-end",
                                    ),
                                ]),
                            ]
                        ),
                    )
                ],
                style={
                    "position": "sticky",
                    "top": "50px"
                },
            ),

        ], width=3),
        dbc.Col([
            dcc.Graph(id='topic_graph'),
            dcc.Graph(id='dominant_topic_graph'),
            dcc.Graph(id='stacked_dominant_topic_graph'),
            dcc.Graph(id='stacked_bar_graph'),
            dcc.Graph(id='coherence_bar'),
            dash_table.DataTable(
                id='topic_table',
                style_cell={'textAlign': 'center'},
            )
        ], width=9, style={'border': '1px solid', 
                           'border-color': 'gray', 
                           'border-radius': '5px',
                           'margin-top': '13px',
                           'margin-bottom': '20px', 
                           'padding': '10px'})
    ])
])


@app.callback(
    [Output('topic_graph', 'figure'),
     Output('dominant_topic_graph', 'figure'),
     Output('stacked_dominant_topic_graph', 'figure'),
     Output('stacked_bar_graph', 'figure'),
     Output('topic_table', 'data')],
     Output('coherence_bar', 'figure'),
     Output('coherence_bar', 'style'),
     
    [Input('update_button', 'n_clicks')],
    [State('num_topics', 'value'),
     State('min_df', 'value'),
     State('term_weight', 'value'),
     State('total_iteraciones', 'value'),
     State('num_top_palabras', 'value'),
     State('upload_data', 'filename'),
     State('toggle_coherence', 'value')]

)


def update_graphs(n_clicks, num_topics, min_df, term_weight, total_iteraciones, num_top_palabras, filename, calculate_coherence):
    
    if filename is None:
        return dash.no_update
  
    print(f'Archivo seleccionado: {filename}')

    global modelo_dtm
    datos = pd.read_csv(f"data/{filename}")
    datos['date'] = pd.to_datetime(datos['date'])
    min_year = datos['date'].dt.year.min()
    max_year = datos['date'].dt.year.max()
    num_years = max_year - min_year + 1
    intervalo_por_año = {year: i for i, year in enumerate(range(min_year, max_year + 1))}
    datos['timepoint'] = datos['date'].dt.year.map(intervalo_por_año)
    num_timepoints = len(intervalo_por_año)
    modelo_dtm = entrenar_modelo_dtm(datos, num_timepoints, num_topics, min_df, term_weight, total_iteraciones)

    topicos_por_año = {year: extraer_topicos(modelo_dtm, timepoint=intervalo_por_año[year], top_n=num_top_palabras) for year in range(min_year, max_year + 1)}
    visualizacion = visualizar_topicos_por_año(topicos_por_año)

    table_data = visualizacion.to_dict('records')
    num_documentos_por_año = datos.groupby(datos['date'].dt.year).size().tolist()

    dominant_topic_counts = count_dominant_topics(modelo_dtm, num_topics, num_documentos_por_intervalo=num_documentos_por_año)
    top_words = {k: [' '.join(topicos_por_año[year][k]) for year in range(min_year, max_year + 1)] for k in range(num_topics)}

    figure_topic = plot_topic_evolution_interac(modelo_dtm, num_topics, num_documentos_por_año, intervalo_por_año, topicos_por_año)
    figure_dominant_topic = plot_dominant_topic_evolution_interac(dominant_topic_counts, intervalo_por_año, topicos_por_año)
    figure_stacked_dominant_topic = plot_dominant_topic_evolution_stacked_interac(dominant_topic_counts, intervalo_por_año, topicos_por_año)
    figure_stacked_bar = plot_stacked_bar_interac(modelo_dtm, num_topics, num_documentos_por_año, intervalo_por_año, topicos_por_año)

    if calculate_coherence:
        print("DEBO CALCULAR COHERENCIA")
        coherence_values = calcular_coherencia(modelo_dtm, datos['cleaned_text'], num_topics, num_top_palabras, num_timepoints)
        print("\nCoherencia entre los tópicos:")
        for i, coherence in enumerate(coherence_values):
            print(f'Coherencia del tópico {i + 1}: {coherence}')

        figure_coherence_bar = plot_coherence(coherence_values, num_topics)
        coherence_style = {'display': 'block'}
    else:
        figure_coherence_bar = go.Figure()  
        coherence_style = {'display': 'none'}

    return [figure_topic, figure_dominant_topic, figure_stacked_dominant_topic, figure_stacked_bar, table_data, figure_coherence_bar, coherence_style]



if __name__ == '__main__':
    app.run_server(debug=True)