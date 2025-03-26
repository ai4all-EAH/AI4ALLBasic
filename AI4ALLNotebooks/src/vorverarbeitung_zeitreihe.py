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

# Zeitachse für die gesamte Datenreihe
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

outlier_widget = widgets.Dropdown(options=['keine', 'Entfernen', 'Log Transformation'], description='Ausreißer:')
scaling_widget = widgets.Dropdown(options=['keine', 'Z-Transformation', 'Min-Max-Skalierung'], description='Normalisierung:')
smoothing_widget = widgets.Dropdown(options=['keine', 'Gleitender Mittelwert 5', 'Gleitender Mittelwert 20','Gaussian'], description='Glättung:')

def update_plot(imputation, outlier_handling, scaling, smoothing, time_range):
    # Erstelle Figure und Achsen
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
    # Bereinige die Achsen, bevor neue Daten gezeichnet werden
    for ax in axs:
        ax.clear()

    data_processed = data_acc.copy()
    original_data = data_processed.copy()

    # Zeitachse für die gesamte Datenreihe
    time = np.linspace(0, len(data_acc)/50, len(data_acc))
    # Umgang mit Ausreißern: Markiere Ausreißer als NaN
    if outlier_handling == 'Entfernen':
        z_scores = np.abs(stats.zscore(data_processed[~data_processed.isnull()]))
        mask = (z_scores < 3)
        data_processed[~data_processed.isnull()] = data_processed[~data_processed.isnull()].where(mask, np.nan)
    elif outlier_handling == 'Log Transformation':
        data_processed = np.log(data_processed - data_processed.min() + 1)
        
    missing_idx = pd.isnull(data_processed)
    
    # Imputation
    if imputation != 'keine':
        if imputation == 'Mittelwert':
            imputer = SimpleImputer(strategy='mean')
        elif imputation == 'Median':
            imputer = SimpleImputer(strategy='median')
        elif imputation == 'kNN':
            all_data_processed = data_acc_prepro.iloc[:,1:4].copy()
            all_data_processed.iloc[:,1] = data_processed
            values = all_data_processed.iloc[:,1:4].values
            est = KNNImputer(n_neighbors=3).fit(values)
            data_processed = pd.Series(est.transform(values)[:,0])
        if imputation in ['Mittelwert', 'Median']:
            data_processed = pd.Series(imputer.fit_transform(data_processed.values.reshape(-1, 1)).ravel())
        elif imputation == 'Lineare Interpolation':
            data_processed.interpolate(method='linear', inplace=True)
        elif imputation == 'LOCF':
            data_processed.ffill(inplace=True)
        elif imputation == 'NOCF':
            data_processed.bfill(inplace=True)
    
    # Skalierung trotz NaNs
    if scaling != 'keine':
        not_nan_mask = ~data_processed.isnull()
        if scaling == 'Z-Transformation':
            scaler = StandardScaler()
        elif scaling == 'Min-Max-Skalierung':
            scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(data_processed[not_nan_mask].values.reshape(-1, 1)).ravel()
        data_processed.loc[not_nan_mask] = scaled_values
    
    # Überprüfe, ob NaNs vorhanden sind, und aktualisiere die Glättungsoptionen
    if data_processed.isnull().any():
        smoothing_widget.value = 'keine'
        smoothing_widget.disabled = True
    else:
        smoothing_widget.disabled = False
    
    # Glättung nur, wenn keine NaNs vorhanden sind und Glättung nicht auf "None" gesetzt ist
    if smoothing != 'keine' and not data_processed.isnull().any():
        if smoothing == 'Gleitender Mittelwert 5':
            data_processed = pd.Series(np.convolve(data_processed, np.ones(5)/5, mode='same'))
        elif smoothing == 'Gleitender Mittelwert 20':
            data_processed = pd.Series(np.convolve(data_processed, np.ones(20)/20, mode='same'))
        elif smoothing == 'Gaussian':
            data_processed = pd.Series(ndimage.gaussian_filter(data_processed, sigma=2))
    
    # Zeitbereich basierend auf dem Slider auswählen
    start_idx, end_idx = time_range
    selected_time = time[start_idx:end_idx+1]
    data_processed = data_processed[start_idx:end_idx+1]
    original_data = original_data[start_idx:end_idx+1]
       
    axs[0].plot(selected_time, original_data, label='Original')
    axs[0].legend()
    
    axs[1].plot(selected_time, data_processed, label='Processed')
    axs[1].legend()
    
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