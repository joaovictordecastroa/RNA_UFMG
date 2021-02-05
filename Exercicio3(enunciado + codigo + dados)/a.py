import numpy as np
import plotly.graph_objects as go
import pandas as pd
import random
from Adaline import Adeline


X = pd.read_csv('./Ex1_x.csv', header=None)
Y = pd.read_csv('./Ex1_y.csv', header=None)
T = pd.read_csv('./Ex1_t.csv', header=None)
print('X')
fig = go.Figure(data=go.Scatter(x=T[0], y=X[0], mode='lines+markers', name='Entradas'))
fig.add_trace(go.Scatter(x=T[0], y=Y[0], mode='lines+markers', name='Saídas'))
fig.update_layout(title='Entradas e Saídas', width=1024, height=516)

test_size = round(T.shape[0]*0.3)

test_vector_x = X[0][:test_size]
test_vector_y = Y[0][:test_size]

fig.show()

# Cálculo da quantidade de épocas para treinamento
epoch = round((T.shape[0])*0.7)

eta = 0.01

print('Adaline')
clf = Adaline(eta, epoch)
clf.training(X[0][:epoch], Y[0][:epoch], [0,0])




fig1 = go.Figure(go.Scatter(x=T[0], y=(X[0]*B + bias), mode='lines+markers', name='Calculado'))
fig1.add_trace(go.Scatter(x=T[0], y=Y[0], mode='lines+markers', name='Original'))

fig1.show()
    
