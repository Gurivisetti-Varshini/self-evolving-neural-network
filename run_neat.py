import neat

# ================= XOR FITNESS =================
def xor_eval(genomes, config):
    xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_outputs = [0, 1, 1, 0]

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0

        for xi, xo in zip(xor_inputs, xor_outputs):
            out = net.activate(xi)[0]
            genome.fitness += 1 - abs(out - xo)


# ================= RUN NEAT (ROBUST) =================
def run_neat(generations, config_path="config-feedforward.txt"):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    best_fitness = []
    mean_fitness = []
    node_counts = []
    conn_counts = []
    species_counts = []
    genomes_by_generation = []

    def per_generation(genomes, config):
        xor_eval(genomes, config)

        fitnesses = [g.fitness for _, g in genomes]
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)

        best_genome = max([g for _, g in genomes], key=lambda g: g.fitness)

        best_fitness.append(best)
        mean_fitness.append(mean)
        node_counts.append(len(best_genome.nodes))
        conn_counts.append(len(best_genome.connections))
        species_counts.append(len(pop.species.species))
        genomes_by_generation.append([g for _, g in genomes])

    winner = pop.run(per_generation, generations)

    return (
        best_fitness,
        mean_fitness,
        node_counts,
        conn_counts,
        species_counts,
        genomes_by_generation,
        winner,
        config
    )
