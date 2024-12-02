import ipywidgets as widgets
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class ColorScheme:
    """Color scheme constants for the visualization."""
    ADELIE: str = '#A71930'
    GENTOO: str = '#006D55'
    CHINSTRAP: str = '#0F204B'
    TEST: str = '#E98300'

@dataclass
class MarkerScheme:
    """Color scheme constants for the visualization."""
    ADELIE: str = 'o'
    GENTOO: str = '*'
    CHINSTRAP: str = 'D'
    TEST: str = 's'

class kNNVisualizer:
    def __init__(self):
        self.df = pd.read_csv('./data/penguins.csv')
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.class_column = 'Pinguinart'
        self.colors = ColorScheme()
        self.markers = MarkerScheme()
        
        self.feature1 = 'Schnabellaenge'
        self.feature2 = 'Schnabelhoehe'
        self.k = 3
        
        self.test_point = pd.DataFrame([{
            self.feature1: self.df[self.feature1].mean(),
            self.feature2: self.df[self.feature2].mean()
        }])
        
        self.prepare_data()
        
    def prepare_data(self) -> None:
        X = self.df[[self.feature1, self.feature2]]
        y = self.df[self.class_column]
        self.X_train, _, self.y_train, _ = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        self.df_train = self.X_train.join(self.y_train)
    
    def update_features(self, feature1: str, feature2: str) -> None:
        self.feature1 = feature1
        self.feature2 = feature2
        self.test_point = pd.DataFrame([{
            self.feature1: self.df[self.feature1].mean(),
            self.feature2: self.df[self.feature2].mean()
        }])
        self.prepare_data()

    def preprocess_data(self, data: pd.DataFrame, 
                       method: str, 
                       reference_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if method == 'keine':
            return data
            
        ref_data = reference_data if reference_data is not None else data
        
        if method == 'Min-Max-Skalierung':
            return (data - ref_data.min()) / (ref_data.max() - ref_data.min())
        elif method == 'Z-Transformation':
            return (data - ref_data.mean()) / ref_data.std()
            
        return data

    def calculate_distances(self, test_pt: pd.DataFrame, 
                          train_data: pd.DataFrame, p: int) -> pd.DataFrame:
        distances = []
        for idx in train_data.index:
            dist = np.sum(np.abs(test_pt.values[0] - train_data.loc[idx]) ** p) ** (1/p)
            distances.append(dist)
            
        return pd.DataFrame(data=distances, index=train_data.index, columns=['Distanz'])

    def plot_knn(self, df_nn: pd.DataFrame, 
                test_pt: pd.DataFrame, 
                k: int, 
                p: int, 
                label: str) -> None:
        
        marker_styles = {
            'Adelie': self.markers.ADELIE,
            'Gentoo': self.markers.GENTOO,
            'Chinstrap': self.markers.CHINSTRAP
        }
        
        plt.figure(figsize=(10, 6))
        
        ax = sns.scatterplot(
            data=self.df_train,
            x=self.feature1,
            y=self.feature2,
            hue=self.class_column,
            style=self.class_column,
            markers=marker_styles,
            palette={
                'Adelie': self.colors.ADELIE,
                'Gentoo': self.colors.GENTOO,
                'Chinstrap': self.colors.CHINSTRAP
            },
            alpha=0.25
        )
        
        plt.plot(
            test_pt[self.feature1],
            test_pt[self.feature2],
            marker=self.markers.TEST,
            color=self.colors.TEST,
            label='Testpunkt',
            markersize=10
        )
        
        sns.scatterplot(
            data=df_nn[0:k],
            x=self.feature1,
            y=self.feature2,
            hue=self.class_column,
            style=self.class_column,
            markers=marker_styles,
            palette={
                'Adelie': self.colors.ADELIE,
                'Gentoo': self.colors.GENTOO,
                'Chinstrap': self.colors.CHINSTRAP
            },
            legend=False,
            s=100
        )
        
        self._plot_distance_lines(df_nn, test_pt, k, p)
        
        ax.legend(title='')
        ax.set_title(f'Vorhersage fuer den Testpunkt: {label}')
        ax.set_axisbelow(True)
        plt.grid(True, zorder=-1.0)
        plt.show()

    def _plot_distance_lines(self, df_nn: pd.DataFrame, 
                           test_pt: pd.DataFrame, 
                           k: int, 
                           p: int) -> None:
        for i in range(k):
            x1 = df_nn[self.feature1].iloc[i]
            y1 = df_nn[self.feature2].iloc[i]
            x2 = test_pt[self.feature1][0]
            y2 = test_pt[self.feature2][0]
            
            if p == 1:  # Manhattan
                plt.plot([x1, x1], [y1, y2], color='gray', linestyle='--', zorder=0)
                plt.plot([x1, x2], [y2, y2], color='gray', linestyle='--', zorder=0)
            else:  # Euclidean
                plt.plot([x1, x2], [y1, y2], color='gray', linestyle='--', zorder=0)

    def create_widget(self) -> None:
        style = {'description_width': 'initial'}
        
        # Error message widget
        error_widget = widgets.HTML(
            value='',
            style={'description_width': 'initial'}
        )
        
        feature1_dropdown = widgets.Dropdown(
            options=self.numeric_columns,
            value=self.feature1,
            description='Merkmal 1:',
            style=style
        )
        
        feature2_dropdown = widgets.Dropdown(
            options=self.numeric_columns,
            value=self.feature2,
            description='Merkmal 2:',
            style=style
        )
        
        feature_select = widgets.VBox([
            error_widget,
            feature1_dropdown,
            feature2_dropdown
        ])
        
        knn_params = widgets.VBox([
            widgets.IntSlider(
                min=1,
                max=260,
                value=self.k,
                description='𝑘:',
                style=style
            ),
            widgets.ToggleButtons(
                options=['Manhattan', 'Euklidisch'],
                value='Euklidisch',
                description='Distanznorm:',
                style=style
            ),
            widgets.ToggleButtons(
                options=['keine', 'Min-Max-Skalierung', 'Z-Transformation'],
                value='keine',
                description='Normalisierung:',
                style=style
            )
        ])
        
        test_point = widgets.VBox([
            widgets.FloatText(
                value=float(self.test_point[self.feature1][0].round(2)),
                description=f'Testpunkt {self.feature1}:',
                style=style,
                decimals=2
            ),
            widgets.FloatText(
                value=float(self.test_point[self.feature2][0].round(2)),
                description=f'Testpunkt {self.feature2}:',
                style=style,
                decimals=2
            )
        ])
        
        tab = widgets.Tab()
        tab.children = [feature_select, knn_params, test_point]
        tab.set_title(0, 'Merkmale')
        tab.set_title(1, '𝑘NN Parameter')
        tab.set_title(2, 'Testpunkt')
        
        display(tab)
        
        out = widgets.Output()
        display(out)
        
        def check_features():
            if feature1_dropdown.value == feature2_dropdown.value:
                error_widget.value = '<p style="color: red;">Bitte wählen Sie zwei unterschiedliche Merkmale aus!</p>'
                return False
            error_widget.value = ''
            return True
        
        def update(change):
            if not check_features():
                return
                
            feature1 = feature1_dropdown.value
            feature2 = feature2_dropdown.value
            k = knn_params.children[0].value
            dist_norm = knn_params.children[1].value
            preprocess = knn_params.children[2].value

            old_feature1 = self.feature1
            old_feature2 = self.feature2
            
            if feature1 != self.feature1 or feature2 != self.feature2:
                self.update_features(feature1, feature2)
                
                test_point.children[0].description = f'Testpunkt {feature1}:'
                test_point.children[1].description = f'Testpunkt {feature2}:'

                if feature1 != old_feature1:
                    test_point.children[0].value = float(self.df[feature1].median())
                if feature2 != old_feature2:
                    test_point.children[1].value = float(self.df[feature2].median())

            test_pt_1 = test_point.children[0].value
            test_pt_2 = test_point.children[1].value
            
            test_pt = pd.DataFrame([{
                self.feature1: test_pt_1,
                self.feature2: test_pt_2
            }])
            
            X_train_processed = self.preprocess_data(self.X_train, preprocess)
            test_pt_processed = self.preprocess_data(test_pt, preprocess, self.X_train)
            
            p = 1 if dist_norm == 'Manhattan' else 2
            distances = self.calculate_distances(test_pt_processed, X_train_processed, p)
            
            distances = distances.sort_values(by=['Distanz'])
            df_nn = distances.join(self.df_train)
            
            counter = Counter(self.y_train[df_nn[0:k].index])
            label = counter.most_common()[0][0]
            
            with out:
                out.clear_output(wait=True)
                self.plot_knn(df_nn, test_pt, k, p, label)
        
        for child in feature_select.children[1:]:  # Skip error widget
            child.observe(update, names='value')
        for child in knn_params.children:
            child.observe(update, names='value')
        for child in test_point.children:
            child.observe(update, names='value')
        
        update(None)

visualizer = kNNVisualizer()
visualizer.create_widget()
