import neat
import numpy as np
import matplotlib.pyplot as plt

CONFIG_FILE = "config-feedforward.txt"
GENERATIONS = 50

fitness_history = []

def eval_genomes(genomes, config):
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0

        xor_inputs = [(0,0), (0,1), (1,0), (1,1)]
        xor_outputs = [0, 1, 1, 0]

        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)[0]
            fitness += (1 - abs(output - xo))

        genome.fitness = fitness

def run_neat(config_file, generations):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genomes, generations)

    for g in range(len(stats.most_fit_genomes)):
        fitness_history.append(stats.most_fit_genomes[g].fitness)

    plot_fitness()
    return winner

def plot_fitness():
    plt.figure(figsize=(8,5))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("NEAT Fitness Evolution")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_neat(CONFIG_FILE, GENERATIONS)
