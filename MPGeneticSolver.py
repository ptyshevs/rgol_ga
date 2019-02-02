from GeneticSolver import GeneticSolver
import multiprocessing as mp
import scipy


def work(solver, Y, delta):
    # this is required for every worker to have different initial seed. Otherwise they inherit it from this thread
    scipy.random.seed()
    return solver.solve(Y, delta)


class MPGeneticSolver:
    def __init__(self, n_proc='auto', *args, **kwargs):
        """
        Multi-process version of Genetic Solver with different initial conditions
        :param n_proc: number of processes to create
        :param args: GeneticSolver arguments (see its documentation for more)
        :param kwargs: GeneticSolver key-value arguments
        """
        if n_proc == 'auto':
            n_proc = mp.cpu_count()
        self.n_proc = n_proc
        self.pool = mp.Pool(mp.cpu_count() if n_proc == 'auto' else n_proc)
        self.args = args
        self.kwargs = kwargs
        self._solvers = None
        if 'fitness_parallel' in self.args or ('fitness_parallel' in self.kwargs and self.kwargs['fitness_parallel']):
            raise ValueError("Fitness function cannot be parallelized in MPGeneticSolver")

    def solve(self, Y, delta, return_all=True):
        """
        Solve RGoL problem
        :param Y: 20x20 array that represents field in stopping condition
        :param delta: number of steps to revert
        :param return_all: if True, returns all of the results from different runners, as well as their scores.
                           If False only solution associated with the best score is returned
        :return: either list of (solution, score) pairs or the best solution (see `return_all`)
        """
        self._solvers = [GeneticSolver(*self.args, **self.kwargs) for _ in range(self.n_proc)]
        tasks = [(solver, Y, delta) for solver in self._solvers]
        results = self.pool.starmap(work, tasks)
        return results if return_all else self.select_best(results)

    @classmethod
    def select_best(cls, solutions):
        """
        Using output of solve method, select the best solution
        :param solutions: list of (solution, score) pairs
        :return: 20x20 array that represents the solution (starting board condition)
        """
        return sorted(solutions, key=lambda x:x[1], reverse=True)[0]


if __name__ == '__main__':
    print(f"Registered number of cores: {mp.cpu_count()}")
