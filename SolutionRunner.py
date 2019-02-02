import pandas as pd
import numpy as np
from MPGeneticSolver import MPGeneticSolver

class SolutionRunner:
  def __init__(self, save_fname='solution.csv', verbosity=0):
    self.save_fname = save_fname
    self.verbosity = verbosity
    self.log = []
    self.running_avg = 0
    self.n = 0
  
  
  def solve_df(self, df, first_n=None, save_to=None):
    solver = MPGeneticSolver(early_stopping=False)
    
    solution_df = pd.DataFrame([], columns=['id', 'score'] + ['start.'+ str(_) for _ in range(1, 401)], dtype=int)
    for col in solution_df.columns:
      solution_df[col] = solution_df[col].astype(np.int32)
    
    self.running_avg = 0
    self.n = 0
    self.log = []
    best, worst = None, None
    for id, (idx, row) in zip(df.index, df.iterrows()):
        delta, Y = row.values[0], row.values[1:].reshape((20, 20)).astype('uint8')
        solution = solver.solve(Y, delta, return_all=False)

        board, score = solution
        flat_board = np.insert(board.ravel(), 0, id)
        flat_board = np.insert(flat_board, 1, int(score * 100))
        solution_df = solution_df.append(pd.Series(flat_board, index=solution_df.columns), ignore_index=True)

        self.log.append((idx, score))
        if best is None or best[1] < score:
            best = (idx, score)
        if worst is None or worst[1] > score:
            worst = (idx, score)
        self.n += 1
        self.running_avg = (self.running_avg * (self.n - 1) + score) / self.n
        if self.verbosity:
          print(f"{idx} is solved with score {score}. Average score: {running_avg}")
        if first_n and idx >= first_n:
          break
    if self.verbosity:
      print("Best score:", best)
      print("Worst score:", worst)
    if save_to is not None:
      solution_df.to_csv(save_to)
    return solution_df