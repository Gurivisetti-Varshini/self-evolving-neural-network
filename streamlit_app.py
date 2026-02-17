import streamlit as st
import neat
import matplotlib.pyplot as plt
import time
import networkx as nx
import os
import neat

# When deployed on Streamlit Cloud, this file lives at:
# /mount/src/self-evolving-neural-network/streamlit_app.py

# So the base directory becomes:
# /mount/src/self-evolving-neural-network/
# BASE_DIR = "/mount/src/self-evolving-neural-network"

# Full deployed path to the NEAT configuration file:
CONFIG_FILE = "config-feedforward.txt "

# Load NEAT configuration
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_FILE
)


# # ================= CONFIG =================
# CONFIG_FILE = "config-feedforward.txt"

# st.set_page_config(layout="wide")
# st.title("🧬 Research Dashboard: Self-Evolving Neural Networks (NEAT)")

# ================= CONTROLS =================
GENERATIONS = st.slider(
    "Select Generations",
    min_value=10,
    max_value=200,
    value=50,
    step=5
)

SPEED_LEVEL = st.slider(
    "Animation Speed (×)",
    min_value=0.01,
    max_value=0.2,
    value=0.05,
)

# Convert speed level to animation delay (inverse relationship)
# Higher speed = smaller delay
ANIM_DELAY = 0.2 / SPEED_LEVEL


# ================= FITNESS FUNCTION =================
def eval_genomes(genomes, config):
    xor_inputs = [(0,0),(0,1),(1,0),(1,1)]
    xor_outputs = [0,1,1,0]

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            out = net.activate(xi)[0]
            genome.fitness += 1 - abs(out - xo)

# ================= AUTO EXPLANATION ENGINE =================
def auto_explain(metric_name, values):
    if len(values) < 5:
        return "Insufficient data to analyze evolutionary behavior."

    start = values[0]
    mid = values[len(values)//2]
    end = values[-1]

    early_gain = mid - start
    late_gain = end - mid
    total_gain = end - start

    if total_gain <= 0:
        trend = "did not show meaningful improvement"
    elif early_gain > late_gain:
        trend = "improved rapidly in early generations and later converged"
    else:
        trend = "showed gradual improvement across generations"

    stability = "stable" if abs(late_gain) < 0.05 else "still evolving"

    return (
        f"The {metric_name.lower()} {trend}. "
        f"Early change was {early_gain:.3f}, late-stage change was {late_gain:.3f}, "
        f"indicating the system is {stability}."
    )

def topology_explain(nodes, conns):
    return (
        f"The network grew by {nodes[-1]-nodes[0]} nodes and "
        f"{conns[-1]-conns[0]} connections. "
        "This confirms performance-driven architecture mutation."
    )

# ================= TOPOLOGY DRAW (SIZE FIXED) =================
def draw_topology(genome, generation):
    G = nx.DiGraph()

    for node in genome.nodes:
        G.add_node(node)

    for (i, o), conn in genome.connections.items():
        if conn.enabled:
            G.add_edge(i, o)

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=350,
        font_size=6,
        node_color="#AED6F1",
        edge_color="gray",
        ax=ax
    )

    ax.set_title(f"Topology – Generation {generation}", fontsize=9)
    ax.axis("off")
    return fig

# ================= RUN NEAT =================
def run_neat(gens):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_FILE
    )

    pop = neat.Population(config)

    best_fitness = []
    mean_fitness = []
    node_counts = []
    conn_counts = []
    species_counts = []
    genomes_by_gen = []

    def per_generation(genomes, config):
        eval_genomes(genomes, config)

        fitnesses = [g.fitness for _, g in genomes]
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)

        best_genome = max([g for _, g in genomes], key=lambda g: g.fitness)

        best_fitness.append(best)
        mean_fitness.append(mean)
        node_counts.append(len(best_genome.nodes))
        conn_counts.append(len(best_genome.connections))
        species_counts.append(len(pop.species.species))
        genomes_by_gen.append([g for _, g in genomes])

    winner = pop.run(per_generation, gens)

    return best_fitness, mean_fitness, node_counts, conn_counts, species_counts, genomes_by_gen, winner

# ================= EXPLANATION WRAPPER =================
def explain(title, text):
    with st.expander(f"📘 {title}"):
        st.write(text)

# ================= ANIMATED PLOT (SIZE FIXED) =================
def animated_plot(y, label):
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=80)
    placeholder = st.empty()

    for i in range(len(y)):
        ax.clear()
        ax.plot(y[:i+1], linewidth=1.8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        placeholder.pyplot(fig, use_container_width=False)
        time.sleep(SPEED_LEVEL)

import os;
local_dir = os.path.dirname(__file__); 
config_path = os.path.join(local_dir, 'config-feedforward')

# ================= RUN =================
if st.button("🚀 Run Evolution"):
    with st.spinner("Running NEAT evolution..."):
        best, mean, nodes, conns, species, genomes_by_gen, winner = run_neat(GENERATIONS)

    st.success("Evolution completed successfully")

    animated_plot(best, "1️. Best Fitness Evolution")
    explain("Best Fitness", auto_explain("Best Fitness", best))

    animated_plot(mean, "2️. Mean Fitness Evolution")
    explain("Mean Fitness", auto_explain("Mean Fitness", mean))

    animated_plot(species, "3️. Species Count")
    explain("Species Count", auto_explain("Species Count", species))

    animated_plot(nodes, "4️. Node Growth")
    explain("Node Growth", auto_explain("Node Growth", nodes))

    animated_plot(conns, "5️. Connection Growth")
    explain("Connection Growth", auto_explain("Connection Growth", conns))

    complexity = [n + c for n, c in zip(nodes, conns)]
    animated_plot(complexity, "6️. Network Complexity")
    explain("Network Complexity", auto_explain("Network Complexity", complexity))

    improvement = [best[i] - best[i-1] if i > 0 else 0 for i in range(len(best))]
    animated_plot(improvement, "7️. Fitness Improvement Rate")
    explain("Fitness Improvement Rate", auto_explain("Fitness Improvement Rate", improvement))

    stability = [abs(best[i] - mean[i]) for i in range(len(best))]
    animated_plot(stability, "8️. Fitness Stability")
    explain("Fitness Stability", auto_explain("Fitness Stability", stability))

    st.subheader("9️. Topology Evolution")
    gen_sel = st.slider("Select Generation", 0, len(genomes_by_gen)-1, 0)

    best_gen = max(
        genomes_by_gen[gen_sel],
        key=lambda g: g.fitness if g.fitness is not None else -1
    )

    st.pyplot(draw_topology(best_gen, gen_sel), use_container_width=False)
    explain("Topology Evolution", topology_explain(nodes, conns))

    st.subheader("10. Automated Research Summary")
    st.markdown(
        f"""
### 🧠 Evolutionary Findings

• Final Best Fitness: **{best[-1]:.3f}**  
• Structural Growth: **+{nodes[-1]-nodes[0]} nodes**, **+{conns[-1]-conns[0]} connections**  
• Maximum Species Observed: **{max(species)}**

These results validate **performance-driven self-evolving neural architectures**,
where complexity increases only when justified by measurable fitness gains.
"""
    )
