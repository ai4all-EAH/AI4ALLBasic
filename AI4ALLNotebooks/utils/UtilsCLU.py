import ipywidgets as widgets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

class WidgetsKMeans:
    def __init__(self, data=None):
        method = self.applyKMeans
        style = {'description_width': 'initial'}

        button_repeat = widgets.ToggleButton(
            value=False,
            description='Wiederholen',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='' # (FontAwesome names without the `fa-` prefix)
        )   

        button_random = widgets.ToggleButton(
            value=False,
            description='Random',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='' # (FontAwesome names without the `fa-` prefix)
        )

        slider_rstate = widgets.IntSlider(
            value=1,
            min=1,
            max=100,
            step=1,
            description="Random State",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )

        slider_it = widgets.IntSlider(
            value=100,
            min=1,
            max=100,
            step=1,
            description="Iterationen",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )

        slider_numclus = widgets.IntSlider(
            value=3,
            min=1,
            max=10,
            step=1,
            description="Number of Clusters",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )

        slider_tol = widgets.FloatSlider(
            value=0.0001,
            min=0.0,
            max=2,
            step=0.0001,
            description="Toleranz",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".4f",
            style=style
        )

        slider_init = widgets.IntSlider(
            value=1,
            min=1,
            max=10,
            step=1,
            description="Wiederholungen",
            tooltip='Anzahl der Wiederholungen des K-Means zur Bestimmung des optimalen Clusterings.',
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )

        widgets.interact(method, 
                 clusters=slider_numclus, 
                 iterations=slider_it, 
                 random_state=slider_rstate, 
                 tolerance= slider_tol,
                 n_init=slider_init,
                 repeat=button_repeat, 
                 randomness=button_random, 
                 X1=widgets.fixed(data))
        
    def applyKMeans(self, clusters, iterations, random_state, tolerance, n_init, repeat, randomness, X1):
        if randomness:
            state = None
        else:
            state = random_state

        algorithm = (KMeans(n_clusters = clusters ,init='random', n_init = n_init ,max_iter=iterations, tol=tolerance,  random_state=state, algorithm='lloyd') )
        algorithm.fit(X1)
        labels1 = algorithm.labels_
        centroids1 = algorithm.cluster_centers_

        plt.figure(1 , figsize = (5 , 5) )


        plt.scatter( x = X1[:,0], y = X1[:,1], c = labels1, s = 1)

        plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 20 , c = 'red' , alpha = 0.5)
        #plt.ylabel('Intensität Gliederschmerzen') , plt.xlabel('Intensität Kopfschmerzen')
        plt.ylabel('Melalgia') , plt.xlabel('Headache')
        plt.show()

def get_linkage_matrix(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix
