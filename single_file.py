import pandas as pd
import numpy as np
from SolutionRunner import SolutionRunner


if __name__ == '__main__':
    df = pd.read_csv('resources/test.csv', index_col='id', skiprows=range(1, 1000))
    sr = SolutionRunner(verbosity=1)
    sr.solve_df(df, 19000)
