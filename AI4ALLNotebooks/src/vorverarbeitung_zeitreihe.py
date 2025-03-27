import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ipywidgets as widgets
from IPython.display import display

data_acc_orig = pd.read_csv('./data/Accelerometer_orig.csv', sep=',', header=0)#, usecols=[0,1])
data_acc_prepro = pd.read_csv('./data/Accelerometer_missing.csv', sep=',', header=0)#, usecols=[0,1])
data_acc = data_acc_prepro.iloc[:,1]

# Zeitachse fÃ¼r die gesamte Datenreihe
time = np.linspace(0, len(data_acc)/50, len(data_acc))
noise =  np.random.normal(0,0.25,len(data_acc))
data_acc = data_acc+noise

# Erstellen des SelectionRangeSlider
time_options = [(f"{t:.2f}", i) for i, t in enumerate(time)]
time_slider = widgets.SelectionRangeSlider(
    options=time_options,
    index=(0, len(time)-1),
    description='Zeitbereich',
    orientation='horizontal',
    layout={'width': '400px'}
)

# Initialisiere Widgets
imputation_widget = widgets.Dropdown(
    options=['keine', 'Mittelwert', 'Median', 'Lineare Interpolation', 'LOCF', 'NOCF', 'kNN'],
    description='Imputation:',
    style={'description_width': 'initial'}
)

outlier_widget = widgets.Dropdown(options=['keine', 'Entfernen', 'Log Transformation'], description='AusreiÃer:')
scaling_widget = widgets.Dropdown(options=['keine', 'Z-Transformation', 'Min-Max-Skalierung'], description='Normalisierung:')
smoothing_widget = widgets.Dropdown(options=['keine', 'Gleitender Mittelwert 5', 'Gleitender Mittelwert 20','Gaussian'], description='GlÃ¤ttung:')

def update_plot(imputation, outlier_handling, scaling, smoothing, time_range):
    # Erstelle Figure und Achsen
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Bereinige die Achsen, bevor neue Daten gezeichnet werden
    for ax in axs:
        ax.clear()

    data_processed = data_acc.copy()
    original_data = data_processed.copy()

    if imputation == 'kNN':
        series_2 = data_acc_prepro.iloc[:, 2].copy()
        series_3 = data_acc_prepro.iloc[:, 3].copy()
        orig_series_2 = data_acc_orig.iloc[:, 2].copy()
        orig_series_3 = data_acc_orig.iloc[:, 3].copy()

    # Zeitachse fÃ¼r die gesamte Datenreihe
    time = np.linspace(0, len(data_acc)/50, len(data_acc))
    
    # Umgang mit AusreiÃern
    if outlier_handling == 'Entfernen':
        for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
            z_scores = np.abs(stats.zscore(data[~data.isnull()]))
            mask = (z_scores < 3)
            data[~data.isnull()] = data[~data.isnull()].where(mask, np.nan)
    elif outlier_handling == 'Log Transformation':
        for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
            data[:] = np.log(data - data.min() + 1)  # Stellen Sie sicher, dass die Operationen auf dem ursprÃ¼nglichen DataFrame durchgefÃ¼hrt werden
    missing_idx = pd.isnull(data_processed)
    
    # Imputation
    if imputation != 'keine':
        if imputation == 'Mittelwert':
            imputer = SimpleImputer(strategy='mean')
            for i, data in enumerate([data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]):
                data[:] = pd.Series(imputer.fit_transform(data.values.reshape(-1, 1)).ravel())
        elif imputation == 'Median':
            imputer = SimpleImputer(strategy='median')
            for i, data in enumerate([data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]):
                data[:] = pd.Series(imputer.fit_transform(data.values.reshape(-1, 1)).ravel())
        elif imputation == 'kNN':
            all_data = data_acc_prepro.iloc[:, 1:4].copy()
            all_data.iloc[:, 0] = data_processed
            values = all_data.values
            est = KNNImputer(n_neighbors=3).fit(values)
            all_data = pd.DataFrame(est.transform(values), columns=all_data.columns)
            data_processed = pd.Series(all_data.iloc[:, 0])
            series_2 = pd.Series(all_data.iloc[:, 1])
            series_3 = pd.Series(all_data.iloc[:, 2])
        elif imputation == 'Lineare Interpolation':
            for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
                data.interpolate(method='linear', inplace=True)
        elif imputation == 'LOCF':
            for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
                data.ffill(inplace=True)
        elif imputation == 'NOCF':
            for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
                data.bfill(inplace=True)
    
    # Skalierung trotz NaNs
    if scaling != 'keine':
        for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
            not_nan_mask = ~data.isnull()
            if scaling == 'Z-Transformation':
                scaler = StandardScaler()
            elif scaling == 'Min-Max-Skalierung':
                scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(data[not_nan_mask].values.reshape(-1, 1)).ravel()
            data.loc[not_nan_mask] = scaled_values
    
    # ÃberprÃ¼fe, ob NaNs vorhanden sind, und aktualisiere die GlÃ¤ttungsoptionen
    if data_processed.isnull().any():
        smoothing_widget.value = 'keine'
        smoothing_widget.disabled = True
    else:
        smoothing_widget.disabled = False
    
    # GlÃ¤ttung nur, wenn keine NaNs vorhanden sind und GlÃ¤ttung nicht auf "None" gesetzt ist
    if smoothing != 'keine' and not data_processed.isnull().any():
        if smoothing == 'Gleitender Mittelwert 5':
            for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
                data[:] = pd.Series(np.convolve(data, np.ones(5)/5, mode='same'))
        elif smoothing == 'Gleitender Mittelwert 20':
            for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
                data[:] = pd.Series(np.convolve(data, np.ones(20)/20, mode='same'))
        elif smoothing == 'Gaussian':
            for data in [data_processed, series_2, series_3] if imputation == 'kNN' else [data_processed]:
                data[:] = pd.Series(ndimage.gaussian_filter(data, sigma=2))
    
    # Zeitbereich basierend auf dem Slider auswÃ¤hlen
    start_idx, end_idx = time_range
    selected_time = time[start_idx:end_idx+1]
    data_processed = data_processed[start_idx:end_idx+1]
    original_data = original_data[start_idx:end_idx+1]
    
    if imputation == 'kNN':
        series_2 = series_2[start_idx:end_idx+1]
        series_3 = series_3[start_idx:end_idx+1]
    
    axs[0].set_title('Originale Zeitreihe')
    axs[1].set_title('Vorverarbeitete Zeitreihe')

    # Zeige die Originaldaten in axs[0]
    axs[0].plot(selected_time, original_data, label='x-Sensor')
    if imputation == 'kNN':
        axs[0].plot(selected_time, orig_series_2, label='y-Sensor')
        axs[0].plot(selected_time, orig_series_3, label='z-Sensor')
    axs[0].legend()

    # Zeige die verarbeiteten Daten in axs[1]
    axs[1].plot(selected_time, data_processed, label='x-Sensor')
    if imputation == 'kNN':
        axs[1].plot(selected_time, series_2, label='y-Sensor')
        axs[1].plot(selected_time, series_3, label='z-Sensor')
    axs[1].legend()

    axs[1].set_xlabel('Zeit in s')
    axs[1].set_ylabel('Beschleunigung in g')

    for idx in data_processed[missing_idx].index:
        if idx >= start_idx and idx <= end_idx:  # Verhindere Indexfehler
            axs[1].plot(selected_time[idx-start_idx:idx+2-start_idx], data_processed[idx-start_idx:idx+2-start_idx], 'r-', linewidth=2, label='Processed')           


    plt.show()
    
ui = widgets.VBox([
    imputation_widget,
    outlier_widget,
    scaling_widget,
    smoothing_widget,
    time_slider
])

out = widgets.interactive_output(update_plot, {
    'imputation': imputation_widget,
    'outlier_handling': outlier_widget,
    'scaling': scaling_widget,
    'smoothing': smoothing_widget,
    'time_range': time_slider
})

display(ui, out)
