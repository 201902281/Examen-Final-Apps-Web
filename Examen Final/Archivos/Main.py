# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import logging
from plotly.subplots import make_subplots
from datetime import datetime
import nbimporter
import plotly.express as px
import yfinance as yf
from sklearn.linear_model import LinearRegression
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import sklearn.datasets
import pandas as pd
import numpy as np
import sklearn.datasets 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
                            silhouette_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
                            silhouette_score, confusion_matrix, ConfusionMatrixDisplay)

    
    
app = dash.Dash()

#app.config.suppress_callback_exceptions = True

logging.getLogger('werkzeug').setLevel(logging.INFO)

df = pd.read_csv("./bank-full.csv", sep = ';')
df_yes = df[df["y"]=="yes"]
df_no = df[df["y"]=="no"]



def generate_specific_plot2():
    df_yes = df[df["y"]=="yes"]
    df_no = df[df["y"]=="no"]

    # Create histograms using Plotly
    histogram_x1 = go.Histogram(x=list(df_yes["duration"]), nbinsx=50, marker=dict(color='black'), opacity=0.7, name='Contratado')
    histogram_x2 = go.Histogram(x=list(df_no["duration"]), nbinsx=50, marker=dict(color='grey'), opacity=0.7, name='No Contratado')
    fig4 = go.Figure(data=[histogram_x1, histogram_x2])
    fig4.layout.xaxis.range = [0, 1000]


    # Update layout
    fig4.update_layout(
        title='Distribucion de la Duracion en Función de la Contratación',
        xaxis=dict(title='Segundos desde el último contacto con cliente'),
        yaxis=dict(title='Frecuencia'),
        showlegend=True
    )


    return fig4


def generate_specific_plot3():
    dum_y = pd.get_dummies(df["y"],prefix="Y") 
    result = pd.concat([df, dum_y], axis=1, join='outer')
    dum_job = pd.get_dummies(df["job"],prefix="job") 
    result = pd.concat([result, dum_job], axis=1, join='outer')
    dum_education = pd.get_dummies(df["education"],prefix="Education") 
    result = pd.concat([result, dum_education], axis=1, join='outer')
    dum_default = pd.get_dummies(df["default"],prefix="default") 
    result = pd.concat([result, dum_default], axis=1, join='outer')
    dum_housing = pd.get_dummies(df["housing"],prefix="housing") 
    result = pd.concat([result, dum_housing], axis=1, join='outer')
    dum_loan = pd.get_dummies(df["loan"],prefix="loan") 
    result = pd.concat([result, dum_loan], axis=1, join='outer')
    df_cleaned = result.iloc[:,[0,5,9,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]]
    X = df_cleaned.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]].copy()
    y = df_cleaned["Y_yes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.3, random_state = 123)

    # Entrenamiento del modelo
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    #Evaluamos la clasificacion
    predictions = lr.predict(X_test)


    print("Confusion matrix")
    cm = confusion_matrix(y_test, predictions)
    print(classification_report(y_test, predictions, target_names=["0","1"]))
    print(cm)

    class_names = ['0', '1']

    # Create annotated heatmap
    heatmap = ff.create_annotated_heatmap(z=cm,
                                          x=["0","1"],
                                          y=["1","0"],
                                          colorscale='Viridis')

    # Update layout
    heatmap.update_layout(title='Confusion Matrix',
                          xaxis=dict(title='Predicted label'),
                          yaxis=dict(title='True label'))

    return heatmap

    
def generate_specific_plot():
    df_sub = df.groupby(["job","y"]).size().unstack(fill_value=0)#Clusters y segements

    # Sample data
    cat = list(df_sub.index)
    No = list(df_sub["no"])
    Yes = list(df_sub["yes"])

    # Create a Plotly figure with stacked bar charts
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=cat, y=No, name='No Contratado', marker_color='mediumseagreen'))
    fig1.add_trace(go.Bar(x=cat, y=Yes, name='Contratado', marker_color='darkorange'))
    fig1.update_layout(
        title='Contratación por tipo de Oficio',
        xaxis=dict(title='Categorias'),
        yaxis=dict(title='Frecuencia'),
        barmode='stack'
    )

    return fig1

def generate_specific_plot4():
    x1 = list(df_yes["previous"])
    x2 = list(df_no["previous"])

    # Create a Plotly figure with box plot
    fig4 = go.Figure()

    fig4.add_trace(go.Box(y=x1, boxpoints='all', name='Contratacion', jitter=0.3, pointpos=-1.8, marker=dict(color='steelblue')))
    fig4.add_trace(go.Box(y=x2, boxpoints='all', name='No Contratacion', jitter=0.3, pointpos=-0.2, marker=dict(color='darkorange')))
    fig4.layout.yaxis.range = [0, 80]
    # Update layout
    fig4.update_layout(
        title='Contactos con el cliente en Función de la Contratación',
        xaxis=dict(title='Distribución del Numero de Contactos con el Cliente', tickvals=[1, 2], ticktext=['', '']),
        yaxis=dict(title='Distribución')
    )

    return fig4

def generate_specific_plot99():
    dum_y = pd.get_dummies(df["y"],prefix="Y") 
    result = pd.concat([df, dum_y], axis=1, join='outer')
    dum_job = pd.get_dummies(df["job"],prefix="job") 
    result = pd.concat([result, dum_job], axis=1, join='outer')
    dum_education = pd.get_dummies(df["education"],prefix="Education") 
    result = pd.concat([result, dum_education], axis=1, join='outer')
    dum_default = pd.get_dummies(df["default"],prefix="default") 
    result = pd.concat([result, dum_default], axis=1, join='outer')
    dum_housing = pd.get_dummies(df["housing"],prefix="housing") 
    result = pd.concat([result, dum_housing], axis=1, join='outer')
    dum_loan = pd.get_dummies(df["loan"],prefix="loan") 
    result = pd.concat([result, dum_loan], axis=1, join='outer')
    df_cleaned = result.iloc[:,[0,5,9,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]]
    X = df_cleaned.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]].copy()
    y = df_cleaned["Y_yes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.3, random_state = 123)

    # Entrenamiento del modelo
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    linear_regression_coef = list(lr.coef_)
    linear_regression_coef = list(linear_regression_coef[0])
    print(linear_regression_coef)
    coef_names = list(X.columns)
    print(coef_names)
    fig = go.Figure(go.Bar(
        x=coef_names,
        y=linear_regression_coef,
        marker_color='mediumseagreen'
    ))

    # Update layout
    fig.update_layout(
        title='Valor Coeficientes',
        xaxis=dict(title='Categorias'),
        yaxis=dict(title='Valor Coeficiente'),
        bargap=0.3,
    )
    return fig

tab0_content = html.Div([
    html.H4(
       children = ["Esperamos que aquellas variables con más cantidad de síes tengan un mayor coeficiente en el modelo de regresión presentado posteriormente. Adicionalmente, se puede observar como conclusión general que la mayoría de las muestras son NO son contrataciones, y que, en métricas agregadas, salvo en los patrones mostrados en los gráficos de barras, no hay variables diferenciales que nos indiquen potencialmente que estén muy correlacionadas con la salida."],
        id = "subtitulo0",
        style ={
            "text-align": "justify",
            "display": "block"
        }
    ),
    html.Div(
        dcc.Graph(
            id='specific_graph',
            figure=generate_specific_plot()  # Set 
        ),
        style={'width': '80%', 'display': 'inline-block'}
    ),
    html.H4(
        children = [""],
        id = "subtitulo1",
        style ={
            "text-align": "justify",
            "display": "block"
        }
    ),
    html.Div(
        dcc.Graph(
            id='specific_graph_2',
            figure=generate_specific_plot2()  # Set 
        ),
        style={'width': '80%', 'display': 'inline-block'}
    ),
    html.H4(
        children = [""],
        id = "subtitulo9",
        style ={
            "text-align": "justify",
            "display": "block"
        }
    ),
    html.Div(
        dcc.Graph(
            id='specific_graph_9',
            figure=generate_specific_plot4()  # Set 
        ),
        style={'width': '80%', 'display': 'inline-block'}
    ),
])

tab1_content = html.Div([
    html.H4(
        children = ["Podemos observar que la regresión lineal construida predice generalmente que un usuario no va a ser contratado. Esta tendencia seguramente se deba al gran numero de muestras de No contrataciones observadas a lo largo del análisis. Por tanto, debe ser considerado como más fiable el análisis descriptivo de las variables más que el valor de los coeficientes del modelo de regresión logística debido a la poca capacidad predictiva del modelo."],
        id = "subtitulo5",
        style ={
            "text-align": "justify",
            "display": "block"
        }
    ),
    html.Div(
        dcc.Graph(
            id='specific_graph_34',
            figure=generate_specific_plot3()  # Set 
        ),
        style={'width': '80%', 'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='specific_graph_99',
            figure=generate_specific_plot99()  # Set 
        ),
        style={'width': '80%', 'display': 'inline-block'}
    ),
])



app.layout = html.Div(
    children= [
        html.H1(
            children = [
                "Analisis de Contrataciones"
            ],
        id = "titulo",
        style = {
            "text-align": "center",
            "text-decoration": "underline",
            "margin-bottom": "20px",
            "padding-top": "20px",
            "height": "50px"
        }
        ),
        dcc.Tabs([
            dcc.Tab(label='Analisis Descriptivo', children=tab0_content, style={'backgroundColor': 'lightblue'},
                    selected_style={'backgroundColor': 'lightblue'}),
            dcc.Tab(label='Modelo Predictivo', children=tab1_content, style={'backgroundColor': 'lightblue'},
                    selected_style={'backgroundColor': 'lightblue'}),
    ])
    ],
    style = {
        "font-family": "Arial"
    }
)



   
        


        

if __name__ == '__main__':
    app.run_server()