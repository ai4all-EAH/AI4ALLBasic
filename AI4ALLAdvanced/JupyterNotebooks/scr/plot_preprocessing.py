from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11})
def plot_imputation(df1, df2, col, start_time=0, end_time=None, title=None, xlabel='Zeit in s', ylabel='Beschleunigung',path='./data/imputation.pdf'):
    '''
    df1: original dataframe without missing data
    df2: dataframe with missing data
    col: column name that contains missing data
    start_time: Startzeitpunkt des Zeitfensters in Sekunden
    end_time: Endzeitpunkt des Zeitfensters in Sekunden
    '''
    # Erstelle die Zeitachse
    time = np.linspace(0, len(df1)/50, len(df1))
    # data_acc_prepro = read_csv('./data/Accelerometer_missing.csv', sep=',', header=0)#, usecols=[0,1])
    # Beschränke die Daten und Zeitachse auf das gewählte Zeitfenster
    if end_time is None:
        end_time = time[-1]
    time_mask = (time >= start_time) & (time <= end_time)
    time_window = time[time_mask]
    df1_window = df1.loc[time_mask]
    df2_window = df2.loc[time_mask]

    df_missing = df2_window.rename(columns={col: 'fehlende Daten und Ausreißer'})

    columns = df_missing.loc[:, 'fehlende Daten und Ausreißer':].columns.tolist()
    subplots_size = len(columns)-2
    fig, ax = plt.subplots(subplots_size+1, 1, sharex=True, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.4)
    if title:
        fig.suptitle(title)

    # Plot original series without highlighting missing data
    ax[0].plot(time_window, df1_window[col], label='Beschleunigung in x-Richtung')  # Verwende `time_window` als X-Werte
    ax[0].set_title('Originale Zeitreihe')
    ax[0].set_ylabel(ylabel)
    # ax[0].set_xlabel(xlabel)

    # Find indices where data is missing in 'Fehlende Daten'
    missing_indices = df_missing['fehlende Daten und Ausreißer'].isnull()

    # Plot 'Fehlende Daten' and highlight missing areas
    l1, = ax[1].plot(time_window, df_missing['fehlende Daten und Ausreißer'], label='Beschleunigung in x-Richtung')  # Verwende `time_window` als X-Werte
    ax[1].set_title('fehlende Daten und Ausreißer')
    ax[1].fill_between(time_window, df_missing['fehlende Daten und Ausreißer'].min(), df_missing['fehlende Daten und Ausreißer'].max(), where=missing_indices, color='red', alpha=0.3)
    # l2, = ax[1].plot(time_window, data_acc_prepro.iloc[:,2], label='Beschleunigung in y-Richtung')  # Verwende `time_window` als X-Werte
    # l3, = ax[1].plot(time_window, data_acc_prepro.iloc[:,3], label='Beschleunigung in z-Richtung')
    ax[1].set_ylabel(ylabel)
    ax[1].set_xlabel(xlabel)
    # ax[1].legend(handles=[l1, l2, l3])

    # Plot each of the other columns and plot missing data in red'
    for i, colname in enumerate(columns[3:], start=2):  # Start from the third subplot
        ax[i].plot(time_window, df_missing[colname], label='Daten mit fehlenden Werten')  # Verwende `time_window` als X-Werte
        ax[i].set_title(colname)
        # Plot missing data in red
        missing_data = df_missing[colname].copy()
        missing_data[~missing_indices] = np.nan
        ax[i].plot(time_window, missing_data, 'r-', label='fehlende Daten und Ausreißer')  # Verwende gefiltertes `time_window` für fehlende Daten
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel(xlabel)
    ax[subplots_size].set_xlabel(xlabel)
    ax[subplots_size].set_ylabel(ylabel)
    fig.savefig(path, bbox_inches='tight', format='pdf')

    plt.show()