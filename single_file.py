#! /home/pn4vin/rgol_ga/ga_env/bin/python3
import pandas as pd
import numpy as np
from SolutionRunner import SolutionRunner


if __name__ == '__main__':
    df = pd.read_csv('resources/test.csv', index_col='id', skiprows=range(1, 20001))
    sr = SolutionRunner(verbosity=1)
    sr.solve_df(df, 8000, 's20k-28k.csv')
