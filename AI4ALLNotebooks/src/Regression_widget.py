import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ipywidgets as widgets


class WidgetRegression:
    def __init__(self, data=None):
        method = self.plotData
        style = {'description_width': 'initial'}

        self.slider_m = widgets.FloatSlider(
            value=1.0,
            min=-2,
            max=2,
            step=0.01,
            description="m",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=style
        )

        self.slider_b = widgets.FloatSlider(
            value=0.2,
            min=-2,
            max=2,
            step=0.01,
            description="b",
            tooltip='Anzahl der Wiederholungen des K-Means zur Bestimmung des optimalen Clusterings.',
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=style
        )

        self.text_block = widgets.HTML(value="<b>Regressionsgleichung:</b> y =" + str(self.slider_b.value))

        self.widget = widgets.interactive_output(method, 
                 {'m': self.slider_m,
                  'b': self.slider_b,
                  'data': widgets.fixed(data)})
        
        # Layout erstellen
        controls = widgets.VBox([self.text_block, self.slider_m, self.slider_b])
        self.layout = widgets.HBox([controls, self.widget])
        
        
    def plotData(self, m, b, data):
        np.random.seed(0)
        X = np.random.rand(100, 1)
        y = X + np.random.randn(100, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        plt.figure(figsize=(5, 4), dpi=100)

        coef = m
        intercept = b
        sum_squared_error = 0.0

        if data is not None:
            plt.scatter(data[:, 0], data[:, 1], color='black', label='Datenpunkte', s=3.0)

            for i in range(len(data[:, 0])):
                y_p = coef * data[i, 0] + intercept
                plt.plot([data[i, 0],data[i, 0]], [data[i, 1], y_p], color='red', linestyle='--', linewidth=0.5)

                # Berechnen der Länge der roten Linie
                line_length = abs(data[i, 1] - y_p)
                
                # Anzeigen der Länge der roten Linie neben dem Datenpunkt
                plt.text(data[i, 0]+0.01, data[i, 1], f'{line_length:.2f}', fontsize=6, color='green')

                # Summe der Fehlerquadrate berechnen
                sum_squared_error += (data[i, 1] - y_p) ** 2


            self.text_block.value = ("<b>Regressionsgleichung:</b> <br> y = {:.2f} {} {:.2f} · x".format( b, '+' if m >= 0 else '-', abs(m)) + \
                                     "<br><b>Summe quadratischer Fehler:</b> <br>{:.4f}".format(sum_squared_error)) + \
                                     "<br><b>Parameter:</b>"
            
        else:
            self.text_block.value = "<b>Regressionsgleichung:</b> <br> y = {:.2f} {} {:.2f} · x".format( b, '+' if m >= 0 else '-', abs(m)) + \
                                    "<br><b>Parameter:</b>"

        # Vorhersagen auf den Testdaten machen
        y_pred = coef * X + intercept




        plt.plot(X, y_pred, color='blue', linewidth=1, label='Gerade')
        plt.xlabel('x', fontsize=8)
        plt.ylabel('y', fontsize=8)
        plt.title('Lineare Regression', fontsize=8)
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()