import os
import typing

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from src.data.data_container import DataContainer


def visualize(client_data: typing.Union[typing.Dict[int, DataContainer], DataContainer]):
    df = pd.DataFrame(columns=['client_id', 'Label', 'data_length', 'x', 'y'])

    for client_id, data in client_data.items():
        df = df.append(
            {'client_id': client_id, 'label': int(np.unique(data.y)), 'data_length': len(data.y), 'x': data.x,
             'y': data.y}, ignore_index=True)
    df.style.hide_index()
    pd.set_option('display.max_columns', None)

    # labels_distribution
    true_labels = ['Child', 'Adult']
    dist = df['label'].value_counts()
    colors = ['mediumturquoise', 'darkorange']
    trace = go.Pie(values=(np.array(dist)), labels=true_labels, textinfo='label+percent+value')
    layout = go.Layout(title='Labels Distribution')
    fig = go.Figure(trace, layout)
    fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    # fig.show()
    save_image(fig, 'labels_distribution')

    # clients_data_distribution
    label = df['label']
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(df['client_id'], df['data_length'], c=df['label'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.title("Clients Data Distribution")
    plt.xlabel('client_id')
    plt.ylabel('data_length')
    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(true_labels)
    plt.savefig('images/clients_data_distribution.png')
    # plt.show()


def save_image(fig_obj, fig_name):
    folder_name = 'images'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    fig_obj.write_image(f'{folder_name}/{fig_name}.png')
