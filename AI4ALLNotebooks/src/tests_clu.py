import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def test_exercise_11(parameters):
    if parameters["dataset_1"]["n_clusters"] == 3 and \
        parameters["dataset_1"]["center_1"]["iterations"] == 3 and \
        parameters["dataset_1"]["center_1"]["working"] == True and \
        parameters["dataset_1"]["center_2"]["iterations"] == 8 and \
        parameters["dataset_1"]["center_2"]["working"] == False:
        print("\033[92mTest 1 passed.\033[0m")
    else:
        print("\033[91mTest 1 failed.\033[0m")

    if parameters["dataset_2"]["n_clusters"] == 3 and \
        parameters["dataset_2"]["center_1"]["iterations"] == 9 and \
        parameters["dataset_2"]["center_1"]["working"] == True and \
        parameters["dataset_2"]["center_2"]["iterations"] == 16 and \
        parameters["dataset_2"]["center_2"]["working"] == False:
        print("\033[92mTest 2 passed.\033[0m")
    else:
        print("\033[91mTest 2 failed.\033[0m")

    if parameters["dataset_3"]["n_clusters"] == 3 and \
        parameters["dataset_3"]["center_1"]["iterations"] == 7 and \
        parameters["dataset_3"]["center_1"]["working"] == True and \
        parameters["dataset_3"]["center_2"]["iterations"] == 4 and \
        parameters["dataset_3"]["center_2"]["working"] == True:
        print("\033[92mTest 3 passed.\033[0m")
    else:
        print("\033[91mTest 3 failed.\033[0m")

    if parameters["dataset_4"]["n_clusters"] == 1 and \
        parameters["dataset_4"]["center_1"]["iterations"] == 1 and \
        parameters["dataset_4"]["center_1"]["working"] == True and \
        parameters["dataset_4"]["center_2"]["iterations"] == 1 and \
        parameters["dataset_4"]["center_2"]["working"] == True:
        print("\033[92mTest 4 passed.\033[0m")
    else:
        print("\033[91mTest 4 failed.\033[0m")

    if parameters["dataset_5"]["n_clusters"] == 6 and \
        parameters["dataset_5"]["center_1"]["iterations"] == 14 and \
        parameters["dataset_5"]["center_1"]["working"] == True and \
        parameters["dataset_5"]["center_2"]["iterations"] == 3 and \
        parameters["dataset_5"]["center_2"]["working"] == True:
        print("\033[92mTest 5 passed.\033[0m")
    else:
        print("\033[91mTest 5 failed.\033[0m")

    if parameters["dataset_6"]["n_clusters"] == 4 and \
        parameters["dataset_6"]["center_1"]["iterations"] == 8 and \
        parameters["dataset_6"]["center_1"]["working"] == True and \
        parameters["dataset_6"]["center_2"]["iterations"] == 3 and \
        parameters["dataset_6"]["center_2"]["working"] == True:
        print("\033[92mTest 6 passed.\033[0m")
    else:
        print("\033[91mTest 6 failed.\033[0m")


def test_exercise_12(parameters):
    if parameters["dataset_1"]["n_clusters"] == 4 and \
        parameters["dataset_1"]["working"] == True:
        print("\033[92mTest 1 passed.\033[0m")
    else:
        print("\033[91mTest 1 failed.\033[0m")

    if parameters["dataset_2"]["n_clusters"] == 2 and \
        parameters["dataset_2"]["working"] == False:
        print("\033[92mTest 2 passed.\033[0m")
    else:
        print("\033[91mTest 2 failed.\033[0m")

    if parameters["dataset_3"]["n_clusters"] == 3 and \
        parameters["dataset_3"]["working"] == False:
        print("\033[92mTest 3 passed.\033[0m")
    else:
        print("\033[91mTest 3 failed.\033[0m")

    if parameters["dataset_4"]["n_clusters"] == 3 and \
        parameters["dataset_4"]["working"] == False:
        print("\033[92mTest 4 passed.\033[0m")
    else:
        print("\033[91mTest 4 failed.\033[0m")

    if parameters["dataset_5"]["n_clusters"] == 4 and \
        parameters["dataset_5"]["working"] == True:
        print("\033[92mTest 5 passed.\033[0m")
    else:
        print("\033[91mTest 5 failed.\033[0m")

    if parameters["dataset_6"]["n_clusters"] == 5 and \
        parameters["dataset_6"]["working"] == False:
        print("\033[92mTest 6 passed.\033[0m")
    else:
        print("\033[91mTest 6 failed.\033[0m")

def test_exercise_21(solution):
    pass