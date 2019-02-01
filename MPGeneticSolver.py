from GeneticSolver import GeneticSolver
import multiprocessing as mp
import scipy


def work(solver, Y, delta):
    # this is required for every worker to have different initial seed. Otherwise they inherit it from this thread
    scipy.random.seed()
    return solver.solve(Y, delta)


class MPGeneticSolver:
    def __init__(self, n_proc='auto', *args, **kwargs):
        if n_proc == 'auto':
            n_proc = mp.cpu_count()
        self.n_proc = n_proc
        self.pool = mp.Pool(mp.cpu_count() if n_proc == 'auto' else n_proc)
        self.args = args
        self.kwargs = kwargs

    def solve(self, Y, delta):
        self.solvers = [(GeneticSolver(*self.args, **self.kwargs), Y, delta) for _ in range(self.n_proc)]
        results = self.pool.starmap(work, self.solvers)
        self.pool.close()
        self.pool.join()
        return results


if __name__ == '__main__':
    print(f"Registered number of cores: {mp.cpu_count()}")
