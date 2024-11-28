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

class kNNVisualizer:
    """Class for visualizing k-Nearest Neighbors classification."""
    
    def __init__(self):
        """Initialize the KNN visualizer with penguin dataset."""
        self.df = pd.read_csv('./data/penguins.csv')
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.class_column = 'Pinguinart'
        self.colors = ColorScheme()
        
        # Default features
        self.feature1 = 'Schnabellaenge'
        self.feature2 = 'Schnabelhoehe'
        self.k = 3
        
        # Initialize test point with mean values
        self.test_point = pd.DataFrame([{
            self.feature1: self.df[self.feature1].mean(),
            self.feature2: self.df[self.feature2].mean()
        }])
        
        self.prepare_data()
        
    def prepare_data(self) -> None:
        """Prepare training data and create initial classifier."""
        X = self.df[[self.feature1, self.feature2]]
        y = self.df[self.class_column]
        self.X_train, _, self.y_train, _ = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        self.df_train = self.X_train.join(self.y_train)
    
    def update_features(self, feature1: str, feature2: str) -> None:
        """Update the features used for visualization."""
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
        """Preprocess the data using the specified method."""
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
        """Calculate distances between test point and training data."""
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
        """Create the kNN visualization plot."""
        plt.figure(figsize=(10, 6))
        
        # Plot training data
        ax = sns.scatterplot(
            data=self.df_train,
            x=self.feature1,
            y=self.feature2,
            hue=self.class_column,
            palette={
                'Adelie': self.colors.ADELIE,
                'Gentoo': self.colors.GENTOO,
                'Chinstrap': self.colors.CHINSTRAP
            },
            alpha=0.25
        )
        
        # Plot test point
        plt.plot(
            test_pt[self.feature1],
            test_pt[self.feature2],
            's',
            color=self.colors.TEST,
            label='Testpunkt',
            markersize=10
        )
        
        # Plot nearest neighbors
        sns.scatterplot(
            data=df_nn[0:k],
            x=self.feature1,
            y=self.feature2,
            hue=self.class_column,
            palette={
                'Adelie': self.colors.ADELIE,
                'Gentoo': self.colors.GENTOO,
                'Chinstrap': self.colors.CHINSTRAP
            },
            legend=False,
            s=100
        )
        
        # Plot distance lines
        self._plot_distance_lines(df_nn, test_pt, k, p)
        
        # Customize plot
        ax.legend(title='')
        ax.set_title(f'Vorhersage fÃ¼r den Testpunkt: {label}')
        ax.set_axisbelow(True)
        plt.grid(True, zorder=-1.0)
        plt.show()

    def _plot_distance_lines(self, df_nn: pd.DataFrame, 
                           test_pt: pd.DataFrame, 
                           k: int, 
                           p: int) -> None:
        """Plot distance lines between test point and neighbors."""
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
        """Create and display the interactive widget with tabs."""
        style = {'description_width': 'initial'}
        
        # Feature selection tab
        feature1_dropdown = widgets.Dropdown(
            options=self.numeric_columns,
            value=self.feature1,
            description='Merkmal 1:',
            style=style
        )
        
        feature2_dropdown = widgets.Dropdown(
            options=[x for x in self.numeric_columns if x != feature1_dropdown.value],
            value=self.feature2,
            description='Merkmal 2:',
            style=style
        )
        
        def update_feature2_options(*args):
            if feature2_dropdown.value == feature1_dropdown.value:
                # Nur updaten wenn wirklich nÃ¶tig
                new_options = [x for x in self.numeric_columns if x != feature1_dropdown.value]
                feature2_dropdown.options = new_options
                feature2_dropdown.value = new_options[0]
        
        feature1_dropdown.observe(update_feature2_options, names='value')
        
        feature_select = widgets.VBox([
            feature1_dropdown,
            feature2_dropdown
        ])
        
        # KNN parameters tab
        knn_params = widgets.VBox([
            widgets.IntSlider(
                min=1,
                max=260,
                value=self.k,
                description='ð:',
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
        
        # Test point tab
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
        
        # Create tabs
        tab = widgets.Tab()
        tab.children = [feature_select, knn_params, test_point]
        tab.set_title(0, 'Merkmale')
        tab.set_title(1, 'ðNN Parameter')
        tab.set_title(2, 'Testpunkt')
        
        # Display widget
        display(tab)
        
        # Create output widget for plot
        out = widgets.Output()
        display(out)
        
        def update(change):
            feature1 = feature_select.children[0].value
            feature2 = feature_select.children[1].value
            k = knn_params.children[0].value
            dist_norm = knn_params.children[1].value
            preprocess = knn_params.children[2].value

            # Store old features for comparison
            old_feature1 = self.feature1
            old_feature2 = self.feature2
            
            # Update features if changed
            if feature1 != self.feature1 or feature2 != self.feature2:
                self.update_features(feature1, feature2)
                
                # Update test point widget labels
                test_point.children[0].description = f'Testpunkt {feature1}:'
                test_point.children[1].description = f'Testpunkt {feature2}:'

                # Set mean values for changed features
                if feature1 != old_feature1:
                    test_point.children[0].value = float(self.df[feature1].median())
                if feature2 != old_feature2:
                    test_point.children[1].value = float(self.df[feature2].median())

            test_pt_1 = test_point.children[0].value
            test_pt_2 = test_point.children[1].value
            
            # Create test point dataframe
            test_pt = pd.DataFrame([{
                self.feature1: test_pt_1,
                self.feature2: test_pt_2
            }])
            
            # Preprocess data
            X_train_processed = self.preprocess_data(self.X_train, preprocess)
            test_pt_processed = self.preprocess_data(test_pt, preprocess, self.X_train)
            
            # Calculate distances and find neighbors
            p = 1 if dist_norm == 'Manhattan' else 2
            distances = self.calculate_distances(test_pt_processed, X_train_processed, p)
            
            # Sort and join with original data
            distances = distances.sort_values(by=['Distanz'])
            df_nn = distances.join(self.df_train)
            
            # Get prediction
            counter = Counter(self.y_train[df_nn[0:k].index])
            label = counter.most_common()[0][0]
            
            # Update plot
            with out:
                out.clear_output(wait=True)
                self.plot_knn(df_nn, test_pt, k, p, label)
        
        # Register callbacks
        for child in feature_select.children:
            child.observe(update, names='value')
        for child in knn_params.children:
            child.observe(update, names='value')
        for child in test_point.children:
            child.observe(update, names='value')
        
        # Initial update
        update(None)


"""Initialize and display the KNN visualization widget."""
visualizer = kNNVisualizer()
visualizer.create_widget()