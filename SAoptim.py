import numpy as np
import random


# Implement SA with convergence data
def sa(fn, bounds, max_iter):
    dimension = len(bounds)
    current = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimension)])
    current_value = fn(current)
    best = current.copy()
    best_value = current_value
    T = 1.0  # Initial temperature
    T_min = 1e-5
    alpha = 0.9  # Cooling rate

    # Store history and convergence data
    history = []
    best_values = []
    while T > T_min and max_iter > 0:
        i = 1
        while i <= 100:
            new_solution = current + np.random.uniform(-1, 1, dimension)

            # Apply bounds
            new_solution = np.clip(new_solution, [b[0] for b in bounds], [b[1] for b in bounds])
            new_value = fn(new_solution)
            delta = new_value - current_value
            if delta < 0 or np.exp(-delta / T) > np.random.rand():
                current = new_solution
                current_value = new_value
            if current_value < best_value:
                best = current.copy()
                best_value = current_value
            i += 1
            history.append([current.copy()])
            best_values.append(best_value)

        T = T * alpha
        max_iter -= 1

    return best, best_value, history, best_values
