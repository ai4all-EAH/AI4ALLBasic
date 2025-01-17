import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def test_exercise_11(parameters):
    if parameters["test1"]["m"] == 1.0 and parameters["test1"]["b"] == 0.4:
        print("\033[92mTest 1 passed.\033[0m")
    else:
        print("\033[91mTest 1 failed.\033[0m")

    if parameters["test2"]["m"] == -1.0 and parameters["test2"]["b"]  == 0.2:
        print("\033[92mTest 2 passed.\033[0m")
    else:
        print("\033[91mTest 2 failed.\033[0m")

    if parameters["test3"]["m"] == 0.0 and parameters["test3"]["b"]  == 0.0:
        print("\033[92mTest 3 passed.\033[0m")
    else:
        print("\033[91mTest 3 failed.\033[0m")

def test_exercise_12(data, parameters):
    # Modell initialisieren mit gegebenen Parametern
    y_pred = parameters["m"] * data[:,0] + parameters["b"]
    # Fehler berechnen
    mse = np.sum((data[:,1] - y_pred) ** 2)

    # Modell initialisieren
    model = LinearRegression()
    model.fit(data[:,0].reshape(-1, 1), data[:,1])

    #X_test = np.random.rand(100, 1)

    # Vorhersagen auf den Testdaten machen
    #y_pred_correct = model.predict(X_test)

    mse_correct = np.sum((data[:,1] - model.predict(data[:,0].reshape(-1, 1))) ** 2)

    mse_correct = np.sum((data[:,1] - model.predict(data[:,0].reshape(-1, 1))) ** 2)

    if np.isclose(mse, mse_correct, atol=0.01):
        print("\033[92mTest passed.\033[0m")
        print("------------------------------------")
        print("Gegebene Gerade: y = {:.2f} {} {:.2f}x".format(parameters["b"], '+' if parameters["m"] >= 0 else '-', abs(parameters["m"])))
        print("Optimale Gerade: y = {:.2f} {} {:.2f}x".format(model.intercept_, '+' if model.coef_[0] >= 0 else '-', abs(model.coef_[0])))
        print("------------------------------------")
        print("Fehler der gegebenen Gerade: {:.4f}".format(mse))
        print("Fehler der optimalen Gerade: {:.4f}".format(mse_correct))
        
        # Plot der Ergebnisse
        plt.scatter(data[:,0], data[:,1], color='black', label='Actual Data')

        # Plotten der optimalen Gerade
        X_test = np.linspace(0, 1, 2)
        y_test = model.predict(X_test.reshape(-1, 1))
        plt.plot(X_test, y_test, color='blue', linewidth=1, label='Optimale Gerade')

        # Plotten der gegebenen Gerade
        y_test2 = parameters["m"] * X_test + parameters["b"]
        plt.plot(X_test, y_test2, color='green', linewidth=1, label='Gegebene Gerade')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Einfache Lineare Regression')
        plt.legend()
        plt.show()
    else:
        print("\033[91mTest failed.\033[0m")
        print("------------------------------------")
        print("Fehler kann durch eine bessere Wahl der Parameter reduziert werden.")

def test_exercise_21(solution):
    if solution["monthly_income"]["prediction"] == True and solution["monthly_income"]["inference"] == True and \
        solution["average_temperature"]["prediction"] == True and solution["average_temperature"]["inference"] == False and \
        solution["distance_to_store"]["prediction"] == False and solution["distance_to_store"]["inference"] == False:
        print("\033[92mTest passed.\033[0m")
        print("\033[93mMonatliches Einkommen in Euro\033[0m: Der Plot zeigt eine lineare Beziehung zwischen dem monatlichen Einkommen und den Ausgaben. Die Ausgaben steigen mit steigendem monatlichen Einkommen. Deshalb ist eine Vorhersage möglich. Zusätzlich ergibt sich aus dem Kontext der Variablen x1 und y2, dass sie in einem kausalen Zusammenhang stehen, weshalb eine Messung des Einflusses möglich ist.")
        print("\033[93mDurchschnittliche monatliche Außentemperatur\033[0m: Der Plot zeigt eine lineare Beziehung zwischen der durchschnittlichen monatlichen Außentemperatur und den Ausgaben. Somit ist auch hier eine Vorhersage möglich. Jedoch stehen diese beiden Größen in keinem kausalen Zusammenhang, weshalb keine Messung des Einflusses möglich ist.")
        print("\033[93mDistanz zum nächsten Supermarkt\033[0m: Hier liegt weder eine lineare Beziehung noch ein kausaler Zusammenhang vor. Deshalb ist weder eine Vorhersage noch eine Messung des Einflusses möglich.")
    else:
        print("\033[91mTest failed.\033[0m")