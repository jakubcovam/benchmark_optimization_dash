import numpy as np
import random


# Implement DE with convergence data
def de(fn, bounds, population_size, max_iter):
    dimension = len(bounds)
    population = [np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimension)]) for _ in
                  range(population_size)]

    # Store history and convergence data
    history = []
    best_values = []
    for t in range(max_iter):
        new_population = []
        for i in range(population_size):
            # Mutation and crossover
            candidates = list(range(0, population_size))
            candidates.remove(i)
            random_indices = random.sample(candidates, 3)
            x1 = population[random_indices[0]]
            x2 = population[random_indices[1]]
            x3 = population[random_indices[2]]
            mutant = x1 + 0.8 * (x2 - x3)
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])

            # Crossover
            cross_points = np.random.rand(dimension) < 0.7
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimension)] = True
            trial = np.where(cross_points, mutant, population[i])

            # Selection
            if fn(trial) < fn(population[i]):
                new_population.append(trial)
            else:
                new_population.append(population[i])
        population = new_population

        # Record positions and best value for visualization
        history.append([ind.copy() for ind in population])
        fitness = [fn(ind) for ind in population]
        best_values.append(min(fitness))

    # Find best solution
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_value = fitness[best_idx]

    return best_solution, best_value, history, best_values
