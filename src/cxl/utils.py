from numpy.typing import NDArray
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import tracemalloc
import pandas as pd
from cxl.graph.graph_utils import create_empty_graph
import itertools


def generate_linear_gaussian(n_samples: int) -> tuple[NDArray, NDArray]:

    p = np.random.randn(n_samples)
    q = np.random.randn(n_samples)
    o = (1 * p) + np.random.randn(n_samples)
    r = p + q + 0.1 * np.random.randn(n_samples)
    s = 0.7 * r + 0.1 * np.random.randn(n_samples)

    ground_truth = create_empty_graph(5)
    ground_truth[0, 2] = 1
    ground_truth[0, 3] = 1
    ground_truth[1, 3] = 1
    ground_truth[3, 4] = 1

    return np.vstack([p, q, o, r, s]).T, ground_truth


def generate_linear_gaussian_p(no_variables, n_samples):
    variables = np.zeros((n_samples, no_variables))

    # Generate the first variable independently
    variables[:, 0] = np.random.randn(n_samples)

    for i in range(1, no_variables):
        if np.random.rand() > 0.5:  # Randomly decide to relate this variable to others
            n_relations = np.random.randint(1, i + 1)
            # Randomly select which variables to use
            indices = np.random.choice(range(i), size=n_relations, replace=False)
            # Generate coefficients for a linear combination of selected variables
            coeffs = np.random.randn(n_relations)
            variables[:, i] = (
                np.dot(variables[:, indices], coeffs) + np.random.randn(n_samples) * 0.1
            )
        else:  # Generate an independent variable
            variables[:, i] = np.random.randn(n_samples)

    return variables


def benchmark(func):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        retval = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took: {end-start} seconds")
        return retval

    return wrapped


def monitor_memory(func):
    pass


def precision(predicted, ground_truth):
    TP = tp(predicted, ground_truth)
    FP = fp(predicted, ground_truth)
    return TP / (TP + FP)


def recall(predicted, ground_truth):
    TP = tp(predicted, ground_truth)
    FN = fn(predicted, ground_truth)
    return TP / (TP + FN)


def f1(predicted, ground_truth):
    rc = recall(predicted, ground_truth)
    pre = precision(predicted, ground_truth)
    return 2 * ((rc * precision) / (rc + pre))


def g_score(predicted, ground_truth):
    TP = tp(predicted, ground_truth)
    FP = fp(predicted, ground_truth)
    FN = fn(predicted, ground_truth)
    return max(0, (TP - FP) / (TP + FN))


def fpr(predicted, ground_truth):
    FP = fp(predicted, ground_truth)
    TN = tn(predicted, ground_truth)
    REV = rev(predicted, ground_truth)
    return (REV + FP) / (TN + FP)


def rev(predicted, ground_truth):
    n = len(predicted)
    count = 0
    for i, j in itertools.combinations(range(n), 2):
        if predicted[i, j] == 1 and predicted[j, i] == 1:
            continue
        if (predicted[i, j] == 1 and ground_truth[j, i] == 1) or (
            predicted[j, i] == 1 and ground_truth[i, j] == 1
        ):
            count += 1
    return count


def tn(predicted, ground_truth):
    n = len(predicted)
    return sum(
        predicted[i, j] == 0 and ground_truth[i, j] == 0
        for i, j in itertools.combinations(range(n), 2)
    )


def tp(predicted, ground_truth):
    n = len(predicted)
    return sum(
        predicted[i, j] == 1 and ground_truth[i, j] == 1
        for i, j in itertools.combinations(range(n), 2)
    )


def fp(predicted, ground_truth):
    n = len(predicted)
    return sum(
        predicted[i, j] == 1 and ground_truth[i, j] == 0
        for i, j in itertools.combinations(range(n), 2)
    )


def fn(predicted, ground_truth):
    n = len(predicted)
    return sum(
        predicted[i, j] == 0 and ground_truth[i, j] == 1
        for i, j in itertools.combinations(range(n), 2)
    )


def pretty_print_graph(graph: NDArray) -> None:
    _, n = graph.shape
    df = pd.DataFrame(
        graph,
    )
    df.index = [f"Node {i}" for i in range(n)]
    df.columns = [f"Node {i}" for i in range(n)]
    print(df)


def draw_graph(graph: NDArray) -> None:
    graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    nx.draw(
        graph,
    )
    plt.show()


def draw_graph_with_labels(graph: NDArray, labels: dict) -> None:
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()


def plot_adjacency_matrices_side_by_side(adj_matrix1, adj_matrix2) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plotting the first adjacency matrix
    axes[0].imshow(adj_matrix1, cmap="Greys", interpolation="nearest")
    axes[0].set_title("Adjacency Matrix 1")

    # Plotting the second adjacency matrix
    axes[1].imshow(adj_matrix2, cmap="Greys", interpolation="nearest")
    axes[1].set_title("Adjacency Matrix 2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mtx = np.array([[0, 1], [1, 0]])
    plot_adjacency_matrices_side_by_side(mtx, mtx)
