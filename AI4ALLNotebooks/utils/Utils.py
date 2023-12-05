import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Bibliothek für die Datenanalyse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import random


class WidgetsGradientDescent:
    def __init__(self, data=None):
        method = self.applyGradientDescent
        style = {'description_width': 'initial'}

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
            value=1,
            min=1,
            max=1000,
            step=1,
            description="Step",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )        
        slider_stepsize = widgets.IntSlider(
            value=1,
            min=1,
            max=236,
            step=1,
            description="Batch Size",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )

        slider_lr = widgets.FloatSlider(
            value=0.01,
            min=0.001,
            max=1.0,
            step=0.001,
            description="Learning Rate",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".3f",
            style=style
        )
        
        slider_elev = widgets.IntSlider(
            value=41,
            min=-90,
            max=90,
            step=1,
            description="Elevation",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )
        
        slider_azim = widgets.IntSlider(
            value=98,
            min=-90,
            max=180,
            step=1,
            description="Azimuth",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )
        
        slider_zoom = widgets.IntSlider(
            value=1,
            min=1,
            max=20,
            step=1,
            description="Zoom",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style=style
        )
        

        widgets.interact(method, 
                 iterations=slider_it, 
                 stepsize=slider_stepsize,
                 random_state=slider_rstate, 
                 lr= slider_lr,
                 elevation=slider_elev,
                 azimuth=slider_azim,
                 zoom=slider_zoom,
                 X=widgets.fixed(data))
        
    def applyGradientDescent(self, iterations, stepsize, random_state, lr,elevation,azimuth, zoom, X):
        # Erstelle ein Figure- und Axes-Objekt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))  # 1 Zeile, 2 Spalten

        x = X[:,0]
        y = X[:,1]
        x_mean = x.mean()
        y_mean = y.mean()
        x = (x-x_mean).reshape(-1,1)
        y = (y-y_mean).reshape(-1,1)

        beta0, beta1, error = self.plotErrorPerIteration(ax1, x, y, iterations, lr, stepsize, random_state)
        self.plotRegressionLine(ax2, x, y, beta0[-1], beta1[-1])
        self.plot3DError(fig, ax3, x, y, elevation,azimuth, beta0, beta1, error, zoom)

        plt.show()


    def plotErrorPerIteration(self, ax, x, y, iterations, lr, stepsize, random_state):
        # Skalieren der Daten
        #scaler = StandardScaler()
        #x_scaled = scaler.fit_transform(x)

        start = random.randint(0, x.shape[0])
        #print(start)

        x_scaled = x

        # Initialisieren des Modells
        sgd_reg = SGDRegressor(random_state=1, max_iter=1, tol=None, eta0=lr, warm_start=True, learning_rate='constant', alpha=0)
        #sgd_reg = SGDRegressor(random_state=np.random.randint(0, 1000), max_iter=1, tol=None, eta0=lr, warm_start=True, learning_rate='constant')

        # Visualisieren Sie die Kosten in jedem Schritt
        n_iterations = iterations#100
        train_errors = []
        beta0 = []
        beta1 = []

        #beta0 = 0
        #beta1 = 0
        #error = 0
        stochastic = True

        counter = 0

        # Erzeugen von zwei zufälligen Zahlen zwischen 0 und 1
        random.seed(random_state)
        coef_init = random.uniform(-1, 3)
        intercept_init = random.uniform(-0.5, 0.5)
        beta0.append(intercept_init)
        beta1.append(coef_init)

        if stochastic:
            for iteration in range(n_iterations):
                start = counter * stepsize#(iteration * stepsize) % x.shape[0]
                end = start + stepsize
                counter += 1

                if end > x.shape[0]:
                    x_scaled = shuffle(x_scaled, random_state=42)
                    y = shuffle(y, random_state=42)
                    counter = 0                
                    start = counter * stepsize
                    end = start + stepsize

                #sgd_reg.fit(x_scaled[start:end].reshape(-1,1), y[start:end].reshape(-1,1))
                sgd_reg.fit(x_scaled[start:end].reshape(-1,1), y[start:end].reshape(-1,1), coef_init=coef_init, intercept_init=intercept_init)

                y_predict = sgd_reg.predict(x_scaled[start:end].reshape(-1,1))

                coef = sgd_reg.coef_
                intercept = sgd_reg.intercept_

                #coef_init = coef
                #intercept_init = intercept

                train_errors.append(mean_squared_error(y[start:end].reshape(-1,1), y_predict))
                beta0.append(intercept[0])
                beta1.append(coef[0])
        else:
            for iteration in range(n_iterations):
                sgd_reg.partial_fit(x_scaled, y)
                y_predict = sgd_reg.predict(x_scaled)

                coef = sgd_reg.coef_
                intercept = sgd_reg.intercept_

                train_errors.append(mean_squared_error(y, y_predict))
                beta0.append(intercept[0])
                beta1.append(coef[0])

                #beta0 = intercept[0]
                #beta1 = coef[0]
                #if iteration == n_iterations - 1:
        
        #print("b0: " + str(beta0) +" - b1: " + str(beta1))


        # Plotten des Trainingsfehlers
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 0.18)

        #ax.plot(np.sqrt(train_errors), "r-+", linewidth=1, label="Training set")
        ax.plot(np.sqrt(train_errors), linewidth=0.5, label="Training set")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE')
        #plt.legend()
        #plt.show()

        ax.text(0.1, 0.8, 'RMSE=' + str(round(train_errors[-1],4)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

        return beta0, beta1, train_errors

    def plotRegressionLine(self, ax, x, y, beta0, beta1):
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=1)),
            ('linear', LinearRegression())
        ])
        model.fit(x, y)

        # Generieren Sie eine Reihe von Werten im Bereich Ihrer Daten
        X_plot = np.linspace(-1, 1, 100).reshape(-1, 1)  # Erzeugt 100 Punkte für eine glatte Linie
        y_plot = beta0 + beta1 * X_plot
        y_plot_MKQ = model.predict(X_plot)  # Vorhersagen mit dem Modell machen

        # Plot die ursprünglichen Datenpunkte
        ax.scatter(x, y, color='lightgrey', edgecolors='black', linewidths=0.1, zorder=2, s=20)

        # Plot das angepasste Polynom
        ax.plot(X_plot, y_plot, color='red', zorder=3)


        # Plot der Regressionsgeraden mit der Methode der kleinsten Quadrate
        ax.plot(X_plot, y_plot_MKQ, color='green', zorder=3, alpha=0.2)

        # Vorhersagen für die gegebenen X-Werte machen
        y_pred = model.predict(x)

        # Titel und Legende hinzufügen
        ax.set_title("PR " + str(1) + ". Grades")

        # Setze die Grenzen der x-Achse von 0 bis 6
        ax.set_xlim(-1, 1)

        # Setze die Grenzen der y-Achse von 0 bis 30
        ax.set_ylim(-1, 1)

        # Achsenbeschriftungen hinzufügen
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Berechnen des Mean Absolute Error (MAE)
        mae = mean_absolute_error(y, y_pred)

        # Berechnen des Mean Squared Error (MSE) und dann des Root Mean Squared Error (RMSE)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        ax.text(0.1, 0.8, 'y=' + str(round(beta0,4)) + '+' + str(round(beta1,4)) + '*x', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    def plot3DError(self, fig, ax, x, y, elevation, azimuth, beta0, beta1, error, zoom):
        x_mean = x.mean()
        y_mean = y.mean()

        x = (x-x_mean).reshape(-1,1)
        y = (y-y_mean).reshape(-1,1)

        reg = LinearRegression().fit(x, y)

        coef = reg.coef_
        intercept = reg.intercept_

        beta0_fix = intercept[0]
        beta1_fix = coef[0,0]
        y = beta0_fix + beta1_fix * x

        intercept_min = -0.5
        intercept_max = 0.5
        intercept_stretch = 0.5
        coefficient_min = -1
        coefficient_max = 3
        coefficient_stretch = 2
        intercept_center = (intercept_min + intercept_max) / 2
        coefficient_center = (coefficient_min + coefficient_max) / 2

        intercept_range = np.linspace(intercept_center - intercept_stretch / zoom, intercept_center + intercept_stretch / zoom, 100)
        coefficient_range = np.linspace(coefficient_center - coefficient_stretch / zoom, coefficient_center + coefficient_stretch / zoom, 100)



        # Wertebereiche für Achsenabschnitt und Koeffizienten

        # intercept_range = np.linspace(-0.5, 0.5, 100)
        # coefficient_range = np.linspace(-1, 3 , 100)

        # Initialisieren der Fehlerwerte Matrix
        error_values = np.zeros((len(intercept_range), len(coefficient_range)))

        # Berechnung der Fehlerwerte für jeden Achsenabschnitt und Koeffizienten
        for i, b0 in enumerate(intercept_range):
            for j, b1 in enumerate(coefficient_range):
                y_pred = b0 + b1 * x
                error_values[i, j] = ((y - y_pred)**2).mean() + 0.1  # Mittlerer quadratischer Fehler

        # Entfernen der Rahmen
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Entfernen der Achsenbeschriftung
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Entfernen der Tick-Markierungen
        ax.tick_params(axis='both', which='both', length=0)

        # Erstellen eines 3D-Oberflächendiagramms
        #fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])



        #print(error_values[0])
        #error_values = np.clip(error_values, 0 , 0.2)

        B0, B1 = np.meshgrid(intercept_range, coefficient_range)
        surf = ax.plot_surface(B0, B1, error_values.T, cmap='coolwarm', alpha=0.75)


        # Erstellen des Scatter-Plots
        #ax.scatter(beta0, beta1, error)
        err_new = []
        for b0, b1 in zip(beta0, beta1):
            yy = b0 + b1 * x
            err = ((y - yy)**2).mean()+0.1
            err_new.append(err)
        ax.scatter(beta0, beta1, err_new, color='green', alpha=1.0, s=1.0)

        # Verbinden Sie aufeinanderfolgende Punkte mit Linien
        for i in range(1, len(beta0)):
            ax.plot([beta0[i-1], beta0[i]], [beta1[i-1], beta1[i]], [err_new[i-1], err_new[i]], color='grey')
            if i == len(beta0) - 1:
                ax.plot([beta0[i-1], beta0[i]], [beta1[i-1], beta1[i]], [err_new[i-1], err_new[i]], color='r')
        

        ax.view_init(elevation, azimuth)

       # Setze die Grenzen der x-Achse von 0 bis 6
        # ax.set_xlim(-0.5, 0.5)
        # ax.set_ylim(-1, 3)
        # ax.set_zlim(0, 0.3)
        ax.set_xlim(intercept_center - intercept_stretch / zoom, intercept_center + intercept_stretch / zoom)
        ax.set_ylim(coefficient_center - coefficient_stretch / zoom, coefficient_center + coefficient_stretch / zoom)
        ax.set_zlim(0, 0.3)

        # Beschriftungen und Titel
        ax.set_xlabel('Intercept')
        ax.set_ylabel('Coefficient')
        ax.set_zlabel('MSE')
        ax.set_title('Error Surface Plot')

        # Farblegende
        #fig.colorbar(surf, shrink=0.5, aspect=5)