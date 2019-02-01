import numpy as np
from tools import generate_field, make_move


class GeneticSolver:
    def __init__(self, population_size=200, n_generations=300, retain_best=0.8, retain_random=0.05, mutate_chance=0.05,
                 verbosity=0, random_state=-1, warm_start=False):
        self.population_size = population_size
        self.n_generations = n_generations
        self.retain_best = retain_best
        self.retain_random = retain_random
        self.mutate_chance = mutate_chance
        self.verbosity = verbosity
        self.random_state = random_state
        self.warm_start = warm_start
        self._population = None
        if random_state != -1:
            np.random.seed(random_state)

    def solve(self, Y, delta, n_generations=-1):
        """

        :param Y: end board (20 x 20 array)
        :return: best solution found
        """
        if not (self._population and self.warm_start):
            self._population = self._generate_population()
        if n_generations != -1:
            self.n_generations = n_generations
        for generation in range(self.n_generations):
            self._population, scores = self.evolve(Y, delta)
            if self.verbosity:
                if generation == 0:
                    print(f"Generation #: best score")
                else:
                    print(f"Generation {generation}: {scores[0]}")
        return self._population[0], scores[0]

    def _generate_population(self, strategy='uniform'):
        if strategy == 'uniform':
            return [generate_field(5) for _ in range(self.population_size)]
        elif strategy == 'covering':
            """ Idea is to cover all the range of possible values for 'density' parameter """
            alive_probabilities = np.linspace(0.01, 0.99, self.population_size)
            return [make_move(np.random.binomial(1, prob, size=(20, 20)), moves=5) for prob in alive_probabilities]
        else:
            raise NotImplementedError(f"{strategy} is not implemented!")

    def evolve(self, target, delta):
        """
        Evolve population
        """
        scores = np.array(self.score_population(self._population, target, delta))
        retain_len = int(len(scores) * self.retain_best)
        sorted_indices = np.argsort(scores)[::-1]
        self._population = [self._population[idx] for idx in sorted_indices]
        best_scores = scores[sorted_indices][:retain_len]
        if self.verbosity > 1:
            print("best scores:", best_scores)
        parents = self._population[:retain_len]
        leftovers = self._population[retain_len:]

        cnt_degenerate = 0
        for gene in leftovers:
            if np.random.rand() < self.retain_random:
                cnt_degenerate += 1
                parents.append(gene)
        if self.verbosity > 1:
            print(f"# of degenerates left: {cnt_degenerate}")

        cnt_mutations = 0
        for gene in parents[1:]:  # mutate everyone expecting for the best candidate
            if np.random.rand() < self.mutate_chance:
                self.mutate(gene)
                cnt_mutations += 1
        if self.verbosity > 1:
            print(f"# of mutations: {cnt_mutations}")

        places_left = self.population_size - retain_len
        children = []
        while len(children) < places_left:
            mom_idx, dad_idx = np.random.randint(0, retain_len - 1, 2)
            if mom_idx != dad_idx:
                child1, child2 = self.crossover(parents[mom_idx], parents[dad_idx])
                children.append(child1)
                if len(children) < places_left:
                    children.append(child2)
        if self.verbosity > 1:
            print(f"# of children: {len(children)}")
        parents.extend(children)
        return parents, best_scores

    @classmethod
    def crossover(cls, mom, dad):
        """
        Take two parents, return two children, shifting half of the genes randomly
        """
        # select_mask = np.random.randint(0, 2, size=(20, 20), dtype='bool')
        select_mask = np.random.binomial(1, 0.5, size=(20, 20)).astype('bool')
        child1, child2 = np.copy(mom), np.copy(dad)
        child1[select_mask] = dad[select_mask]
        child2[select_mask] = mom[select_mask]
        return child1, child2

    @classmethod
    def mutate(cls, field):
        """
        Inplace mutation of the provided field
        """
        a = np.random.binomial(1, 0.1, size=(20, 20)).astype('bool')
        field[a] += 1
        field[a] %= 2
        return field

    @classmethod
    def fitness(cls, start_field, end_field, delta):
        candidate = make_move(start_field, moves=delta)
        return (candidate == end_field).sum() / 400

    @classmethod
    def score_population(cls, population, target, delta):
        return [cls.fitness(gene, target, delta) for gene in population]

if __name__ == '__main__':
    print(GeneticSolver.fitness())
