import ipywidgets as widgets
import pandas as pd
import numpy as np
import matplotlib.markers as markers
from collections import Counter
import seaborn as sns
from matplotlib import pyplot as plt
from ipywidgets import *
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier

ETIT = '#A71930'
MTBT = '#006D55'
GW = '#0F204B'
BW = '#E98300'
VB = '#0039A6'
WI = '#69BE28'

marker = markers.MarkerStyle(marker='o', fillstyle='none')
class_to_color = {klasse1: MTBT, klasse2: ETIT}
cmap_light = ListedColormap([class_to_color[class_name] for class_name in class_to_color])
#cmap_light = ListedColormap([MTBT, ETIT])
cmap_bold = [ETIT, MTBT]

def visualise_kNN_widgets(df_nn, df_train, p, k, label, test_pt, clf, Vorverarbeitung):

    plt.subplots(figsize=(8,4))

    # feature_1, feature_2 = np.meshgrid(
    # np.linspace(X_train[merkmal1].min(), X_train[merkmal1].max()),
    # np.linspace(X_train[merkmal2].min(), X_train[merkmal2].max())
    # )
    # grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    #
    #
    #
    # tree = KNeighborsClassifier().fit(X_train, Y_train)
    # y_pred = np.reshape(tree.predict(grid), feature_1.shape)
    # y_pred[y_pred==klasse1] = 0
    # y_pred[y_pred==klasse2] = 1
    # # feature_1, feature_2 = np.meshgrid(
    # # np.linspace(df_train[merkmal1].min(), df_train[merkmal1].max()),
    # # np.linspace(df_train[merkmal2].min(), df_train[merkmal2].max())
    # # )
    # display = DecisionBoundaryDisplay(
    #     xx0=feature_1, xx1=feature_2, response=y_pred,
    #     # cmap=cmap_light,
    #     # ax=ax,
    #     # response_method="predict",
    #     # plot_method="pcolormesh",
    #     # xlabel=merkmal1,
    #     # ylabel=merkmal2,
    #     # shading="auto",
    #     # alpha=0.2
    # )
    # display.plot()

    ax = sns.scatterplot(data=df_train, x=merkmal1, y=merkmal2, hue=klasse, palette={klasse1: ETIT, klasse2: MTBT}, alpha=0.25)
    plt.plot(test_pt[merkmal1], test_pt[merkmal2], 's', color=BW, label='Testpunkt')
    sns.scatterplot(data=df_nn[0:k], x=merkmal1, y=merkmal2, hue=klasse, palette={klasse1: ETIT, klasse2: MTBT}, legend = False)
    ax.legend(title='')


   # if Vorverarbeitung == 'keine':
   #    DecisionBoundaryDisplay.from_estimator(
   #         clf,
   #         df_train[[merkmal1, merkmal2]],
   #         cmap=cmap_light,
   #         ax=ax,
   #         response_method="predict",
   #         plot_method="pcolormesh",
   #         xlabel=merkmal1,
   #         ylabel=merkmal2,
   #         shading="auto",
   #         alpha=0.2
   #     )

    if p == 1:
        for i in range(k):
            x1 = df_nn[merkmal1].iloc[i]
            y1 = df_nn[merkmal2].iloc[i]
            x2 = test_pt[merkmal1][0]
            y2 = test_pt[merkmal2][0]
            plt.plot([x1, x2], [y1, y2], color='black', zorder=0)
    elif p == 2:
        for i in range(k):
            x1 = df_nn[merkmal1].iloc[i]
            y1 = df_nn[merkmal2].iloc[i]
            x2 = test_pt[merkmal1][0]
            y2 = test_pt[merkmal2][0]
            plt.plot([x1, x2], [y1, y1], color='black', zorder=0)
            plt.plot([x2, x2], [y1, y2], color='black', zorder=0)

    ax.set_title('Vorhersage fuer den Testpunkt: \n ' + label)
    ax.set_axisbelow(True)
    plt.rc('axes', axisbelow=True)
    plt.grid(which='major', zorder=-1.0)
    # def onclick(event):
    #     test_pt_value1 = event.xdata
    #     test_pt_value2 = event.ydata
    sns.set_style("whitegrid")
    # f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def nearest_neighbour_widget(k: int, Distanznorm: str, Testpunkt_Merkmal1, Testpunkt_Merkmal2, Vorverarbeitung: str):
    test_pt = pd.DataFrame([{merkmal1: Testpunkt_Merkmal1, merkmal2: Testpunkt_Merkmal2}])
    # test_pt[merkmal1][0] = Testpunkt_Merkmal1
    # test_pt[merkmal2][0] = Testpunkt_Merkmal2
    # Split des Datensatzes in Merkmale und Label
    # if len(df_reduced) > 1250:
    #     remove_n = len(df_reduced)-1000
    #     drop_indices = np.random.choice(df_reduced.index, remove_n, replace=False)
    #     df_reduced = df_reduced.drop(drop_indices)

    X = df_reduced.drop(columns=[klasse]) # X enthält alle Merkmale für alle Datenpunkte
    Y = df_reduced[klasse] # y enthält alle Label für alle Datenpunkte
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
    df_train = X_train.join(Y_train)
    if Vorverarbeitung == 'Normalisierung':
        # df[merkmal1] = (df_start[merkmal1]-df_start[merkmal1].mean())/df_start[merkmal1].std()
        # df[merkmal2] = (df_start[merkmal2]-df_start[merkmal2].mean())/df_start[merkmal2].std()
        test_pt_changed = (test_pt-X_train.min())/(X_train.max()- X_train.min())
        X_train = (X_train-X_train.min())/(X_train.max()- X_train.min())
        # Split des Datensatzes in Trainings- und Testdaten
    elif Vorverarbeitung == 'Standardisierung':
        test_pt_changed = (test_pt-X_train.mean())/X_train.std()
        X_train = (X_train-X_train.mean())/X_train.std()
    else:
        test_pt_changed = test_pt
    if Distanznorm == 'Euklidisch': p = 1
    if Distanznorm == 'Manhatten': p = 2

    clf = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=p)
    clf.fit(X_train, Y_train)


    # Berechnung der Distanz zwischen den Punkten a und b
    def minkowski_distanz(a, b, p=1):
        # Speichern der Dimensionen (Anzahl an Merkmalen) von Punkt a
        dimension = len(a)
        # Initalisiere die Variabel distanz auf 0
        distanz = 0
        # Berechnung der Minkoswki Distanz mithilfe des festgelegten Parameters p
        for i in range(dimension):
            distanz = distanz + abs(a[i] - b[i]) ** p
            distanz = distanz ** (1 / p)
        return distanz

    # Berechnung der Distanzen zwischen dem Testpunkt test_pt und allen anderen Trainingspunkten X
    distanzen = []
    for j in X_train.index:
        distanzen.append(minkowski_distanz(test_pt_changed.values[0], X_train.loc[j]))
    df_dists = pd.DataFrame(data=distanzen, index=X_train.index, columns=['Distanz'])

    # Finden der k-nächsten Nachbarn
    df_dists = df_dists.sort_values(by=['Distanz'], axis=0)  # Sortieren der k-nächsten Distanzen nach Größe
    df_nn = df_dists.join(df_reduced)  # Integrieren der Daten aus dem ursprünglichen Dataframe zu den Distanzen
    counter = Counter(Y_train[df_nn[0:k].index])
    label = counter.most_common()[0][0]
    visualise_kNN_widgets(df_nn, df_train, p, k, label, test_pt, clf, Vorverarbeitung)

# nearest_neighbour(Distanznorm='Euklidisch', Vorverarbeitung=True, Testpunkt_Merkmal1=60, Testpunkt_Merkmal2=150, k=3)
style = {'description_width': 'initial'}
min1 = min(df_reduced[merkmal1])#[merkmal1][0]-10
max1 = max(df_reduced[merkmal1])#test_punkt[merkmal1][0]+10
min2 = min(df_reduced[merkmal2])#test_punkt[merkmal2][0]-10
max2 = max(df_reduced[merkmal2])#test_punkt[merkmal2][0]+10

widgets.interact(nearest_neighbour_widget, k=widgets.IntSlider(min=1, max=300, value=k), Distanznorm=widgets.ToggleButtons(options=['Manhatten', 'Euklidisch'], value='Euklidisch'), Testpunkt_Merkmal1=widgets.BoundedIntText(min=min1,max=max1,value=test_punkt[merkmal1][0], style=style, description='Testpunkt '+merkmal1), Testpunkt_Merkmal2=widgets.BoundedIntText(min=min2,max=max2,value=test_punkt[merkmal2][0], style=style, description='Testpunkt '+merkmal2), Vorverarbeitung=widgets.ToggleButtons(options=['keine', 'Standardisierung', 'Normalisierung'], value='keine', style=style))