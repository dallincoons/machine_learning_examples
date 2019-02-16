import pandas as pd
from car_data import CarData

def main():
    data = pd.read_csv('./breast-cancer.csv', names=['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])
    CarData(data).transform()


if __name__ == "__main__":
    main()
