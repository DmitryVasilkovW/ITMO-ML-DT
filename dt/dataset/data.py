import pandas as pd

url = "../../data/tic-tac-toe.data"


def get_data():
    data = pd.read_csv(url, header=None)

    data.columns = [
        'top-left-square', 'top-middle-square', 'top-right-square',
        'middle-left-square', 'middle-middle-square', 'middle-right-square',
        'bottom-left-square', 'bottom-middle-square', 'bottom-right-square',
        'Class'
    ]

    pd.set_option('future.no_silent_downcasting', True)
    data.replace({'x': 1, 'o': 0, 'b': -1}, inplace=True)
    data = data.infer_objects(copy=False)

    data['Class'] = data['Class'].apply(lambda x: 1 if x == 'positive' else 0)

    return data
