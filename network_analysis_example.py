"""
Prototype script for network analysis of diplomatic discourse.

This script demonstrates how to build a directed network from dyad-level
diplomatic discourse features and compute basic network statistics.

Prerequisites:
    - pandas
    - networkx
    - matplotlib (for plotting)
    - CSV file with weekly dyad features (e.g., panel_diplomacy_gdelt_week_2019_2024.csv).

The script loads the processed panel, sums tone and frame ratios over a given period,
constructs a directed graph with nodes as countries and edge weights representing
the average tone or frame intensity from sender to receiver, and computes centrality metrics.
It then visualises the network.

Note: this script is a prototype and should be adapted to your specific data schema.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def build_network(panel_path: str, metric: str = "frame_security", start_year: int = 2022, end_year: int = 2024) -> nx.DiGraph:
    """Build a directed network from dyadic panel data.

    Args:
        panel_path: Path to the processed panel CSV containing columns 'src', 'dst', 'week' and the metric.
        metric: Column name to aggregate as edge weight (e.g., 'frame_security', 'tone_positive').
        start_year: Start of the period to consider.
        end_year: End of the period to consider (inclusive).

    Returns:
        A directed graph with nodes for each country and weighted edges from src to dst.
    """
    # Load data
    df = pd.read_csv(panel_path)
    if 'week' not in df.columns:
        raise ValueError("DataFrame must include a 'week' column with dates.")
    # Parse dates and filter by period
    df['week'] = pd.to_datetime(df['week'])
    mask = (df['week'].dt.year >= start_year) & (df['week'].dt.year <= end_year)
    df_period = df.loc[mask].copy()
    # Ensure required columns exist
    for col in ['src', 'dst', metric]:
        if col not in df_period.columns:
            raise ValueError(f"Column '{col}' not found in the data.")
    # Aggregate metric by dyad
    agg = df_period.groupby(['src', 'dst'])[metric].mean().reset_index()
    # Build directed graph
    G = nx.DiGraph()
    for _, row in agg.iterrows():
        src, dst, weight = row['src'], row['dst'], row[metric]
        if pd.notna(weight):
            # Only add edges with positive weight
            if weight > 0:
                G.add_edge(src, dst, weight=weight)
    return G


def compute_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """Compute basic centrality metrics for each node.

    Returns a DataFrame with in-degree, out-degree and betweenness centrality.
    """
    centrality = {
        'node': list(G.nodes()),
        'in_degree': [G.in_degree(n, weight='weight') for n in G.nodes()],
        'out_degree': [G.out_degree(n, weight='weight') for n in G.nodes()],
        'betweenness': list(nx.betweenness_centrality(G, weight='weight').values())
    }
    return pd.DataFrame(centrality)


def plot_network(G: nx.DiGraph, metric: str) -> None:
    """Visualise the directed network using a spring layout.

    Node sizes are proportional to out-degree; edge widths reflect the weight.
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty; nothing to plot.")
        return
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    node_sizes = [300 + 200 * G.out_degree(n, weight='weight') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', edgecolors='k')
    nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights], alpha=0.6, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f'Diplomatic Interaction Network ({metric})')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Example usage:
    # Replace with the actual path to your processed panel CSV
    panel_path = 'data_processed/panel_diplomacy_gdelt_week_2019_2024.csv'
    metric = 'frame_security'
    G = build_network(panel_path, metric=metric)
    print(f'Number of nodes: {G.number_of_nodes()}, number of edges: {G.number_of_edges()}')
    # Compute centrality metrics
    df_centrality = compute_centrality(G)
    print(df_centrality.head())
    # Plot the network
    plot_network(G, metric=metric)