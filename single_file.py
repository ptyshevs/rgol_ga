#! /home/ptyshevs/rgol_ga/ga_env/bin/python3
import pandas as pd
import numpy as np
from SolutionRunner import SolutionRunner


if __name__ == '__main__':
    df = pd.read_csv('resources/test.csv', index_col='id', skiprows=range(1, 10001))
    sr = SolutionRunner(verbosity=1)
    sr.solve_df(df, 10000, 's10k-20k.csv')
