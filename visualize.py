import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_fitness(best, mean):
    fig, ax = plt.subplots()
    ax.plot(best, label="Best Fitness")
    ax.plot(mean, label="Mean Fitness")
    ax.set_title("Fitness Evolution")
    ax.legend()
    return fig

def plot_improvement_rate(best):
    rate = np.diff(best)
    fig, ax = plt.subplots()
    ax.plot(rate)
    ax.set_title("Fitness Improvement Rate")
    return fig

def plot_complexity(nodes, conns):
    fig, ax = plt.subplots()
    ax.plot(nodes, label="Nodes")
    ax.plot(conns, label="Connections")
    ax.set_title("Network Complexity Growth")
    ax.legend()
    return fig

def plot_pareto(conns, best):
    fig, ax = plt.subplots()
    ax.scatter(conns, best)
    ax.set_title("Fitness vs Complexity (Pareto Front)")
    ax.set_xlabel("Connections")
    ax.set_ylabel("Fitness")
    return fig

def plot_species(species):
    fig, ax = plt.subplots()
    ax.plot(species)
    ax.set_title("Species Count Over Generations")
    return fig

def plot_diversity_heatmap(best):
    data = np.array(best).reshape(-1,1)
    fig, ax = plt.subplots()
    sns.heatmap(data, ax=ax, cmap="viridis")
    ax.set_title("Diversity Heatmap")
    return fig
