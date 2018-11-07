import os
import sys
from functools import wraps, reduce


def call_function_get_frame(func, *args, **kwargs):
    """
    Calls the function *func* with the specified arguments and keyword
    arguments and snatches its local frame before it actually executes.
    """

    frame = None
    trace = sys.gettrace()
    def snatch_locals(_frame, name, arg):
        nonlocal frame
        if frame is None and name == 'call':
            frame = _frame
            sys.settrace(trace)
        return trace
    sys.settrace(snatch_locals)
    try:
        result = func(*args, **kwargs)
    finally:
        sys.settrace(trace)
    return frame, result


def update_globals(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        frame, result = call_function_get_frame(func, *args, **kwargs)
        f_locals = frame.f_locals
        # print(f_locals)
        globals().update(f_locals)
        return result
    return wrapper


@update_globals
def import_all(env='local', error_handler=None):
    if env == 'colab':
        from google.colab import drive
        drive.mount('/content/gdrive')

    exit_flag = False
    while not exit_flag:
        try:
            import pymorphy2
            import numpy as np
            import pandas as pd
            import seaborn as sns

            import requests

            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

            from tqdm import tqdm
            tqdm.pandas()

            import ds

            import matplotlib.pyplot as plt

            exit_flag = True
        except ModuleNotFoundError as e:
            print(e.name)
            if error_handler:
                error_handler(e)
            else:
                break
    del exit_flag, error_handler
    return locals()


if __name__ == '__main__':
    import_all()
    print(globals())
