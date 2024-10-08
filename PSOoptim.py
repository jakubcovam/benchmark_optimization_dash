import numpy as np
import random


# Particle class for PSO
class Particle:
    def __init__(self, bounds, dimension):
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimension)])
        self.velocity = np.zeros(dimension)
        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')

    def update_velocity(self, gbest_position):
        w = 0.5  # Inertia weight
        c1 = c2 = 1.0  # Cognitive and social constants
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        # Apply bounds
        for i in range(len(bounds)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])


# Implement PSO with convergence data
def pso(fn, bounds, num_particles, max_iter):
    dimension = len(bounds)

    # Initialize particles
    particles = [Particle(bounds, dimension) for _ in range(num_particles)]

    # Initialize global best
    gbest_position = particles[0].position.copy()
    gbest_value = fn(gbest_position)

    # Store history and convergence data
    history = []
    best_values = []
    for t in range(max_iter):
        for particle in particles:
            fitness_candidate = fn(particle.position)

            # Update personal best
            if fitness_candidate < particle.pbest_value:
                particle.pbest_value = fitness_candidate
                particle.pbest_position = particle.position.copy()

            # Update global best
            if fitness_candidate < gbest_value:
                gbest_value = fitness_candidate
                gbest_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(gbest_position)
            particle.update_position(bounds)

        # Record positions and best value for visualization
        history.append([particle.position.copy() for particle in particles])
        best_values.append(gbest_value)

    return gbest_position, gbest_value, history, best_values

