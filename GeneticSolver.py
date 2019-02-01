import numpy as np
from tools import generate_field, make_move


class GeneticSolver:
    def __init__(self, population_size=200, n_generations=300, retain_best=0.8, retain_random=0.05, mutate_chance=0.05,
                 verbosity=0, random_state=-1, warm_start=False, early_stopping=True, patience=20,
                 initialization_strategy='uniform'):
        """

        :param population_size: number of individual candidate solutions
        :param n_generations: number of generations
        :param retain_best: percentage of best candidates to select into the next generation
        :param retain_random: probability of selecting sub-optimal candidate into the next generation
        :param mutate_chance: candidate mutation chance
        :param verbosity: level of verbosity (0 - quiet, 1 - evolution information, 2 - spamming like in 2003)
        :param random_state: if specified, initializes seed with this value
        :param warm_start: if True, initial population generation step is omitted, allowing for continuing training
        :param early_stopping: if True, evolution will stop if top-10 candidates are not changing for several generations
        :param patience: number of generations to wait for best solution change when <early_stopping>
        :param initialization_strategy: initial population generation rule: 'uniform' or 'covering'
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.retain_best = retain_best
        self.retain_random = retain_random
        self.mutate_chance = mutate_chance
        self.verbosity = verbosity
        self.random_state = random_state
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.patience = patience
        self.initialization_strategy = initialization_strategy

        self._population = None
        if random_state != -1:
            np.random.seed(random_state)

    def solve(self, Y, delta, n_generations=-1):
        """

        :param Y: 20x20 array that represents field in stopping condition
        :param delta: number of steps to revert
        :param n_generations: number of evolution generations. Overrides initialization value if specified
        :return: 20x20 array that represents the best start field found and associated fitness value
        """
        if not (self._population and self.warm_start):
            self._population = self._generate_population()
        if n_generations != -1:
            self.n_generations = n_generations
        scores = np.zeros(len(self._population))
        prev_scores = np.zeros(len(self._population))
        cnt_no_change_in_scores = 0
        for generation in range(self.n_generations):
            self._population, scores = self.evolve(Y, delta)
            if np.isclose(prev_scores[:10], scores[:10]).all():
                cnt_no_change_in_scores += 1
            else:
                cnt_no_change_in_scores = 0
                prev_scores = scores
            if self.verbosity:
                if generation == 0:
                    print(f"Generation #: best score")
                else:
                    print(f"Generation {generation}: {scores[0]}")
            if self.early_stopping and cnt_no_change_in_scores >= self.patience:
                if self.verbosity:
                    print(f"Early stopping on generation {generation} with best score {scores[0]}")
                break
        return self._population[0], scores[0]

    def _generate_population(self):
        """
        Generating initial population of individual solutions

        Regardless of strategy, we make 5 initial "warming" steps to make distribution closer to the problem.

        Strategies description:

            * Uniform: each cell has equal probability of being initialized as alive or dead. This will introduce no
                       prior information at all
            * Covering: Each individual is generated with it's own probability of having each cell 'alive'. This gives
                       on average higher initial fitness score, but has no observed effect on long-term behavior
        :return: initial population as a list of 20x20 arrays
        """
        if self.initialization_strategy == 'uniform':
            return [generate_field(5) for _ in range(self.population_size)]
        elif self.initialization_strategy == 'covering':
            """ Idea is to cover all the range of possible values for 'density' parameter """
            alive_probabilities = np.linspace(0.01, 0.99, self.population_size)
            return [make_move(np.random.binomial(1, prob, size=(20, 20)), moves=5) for prob in alive_probabilities]
        else:
            raise NotImplementedError(f"{self.initialization_strategy} is not implemented!")

    def evolve(self, Y, delta):
        """
        Evolution step
        :param Y: 20x20 array that represents field in stopping condition
        :param delta: number of steps to revert
        :return: new generation of the same size along with scores of the best retained individuals
        """
        scores = np.array(self.score_population(self._population, Y, delta))
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
        Take two parents, return two children, interchanging half of the allels of each parent randomly
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
        """
        Calculate fitness for particular candidate (start configuration of the field)
        :param start_field: candidate (start configuration)
        :param end_field: target (stop configuration)
        :param delta: number of steps to proceed before comparing to stop configuration
        :return: value in range [0, 1] that indicates fractions of cells that match their state
        """
        candidate = make_move(start_field, moves=delta)
        return (candidate == end_field).sum() / 400

    @classmethod
    def score_population(cls, population, Y, delta):
        """
        Apply fitness function for each gene in a population
        :param population: list of candidate solutions
        :param Y: 20x20 array that represents field in stopping condition
        :param delta: number of steps to revert
        :return: list of scores for each solution
        """
        return [cls.fitness(gene, Y, delta) for gene in population]

if __name__ == '__main__':
    print(GeneticSolver.fitness())
