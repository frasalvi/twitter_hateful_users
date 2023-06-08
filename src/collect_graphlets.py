import os
import os.path as osp
import json

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = "data/"
if not osp.exists(BASE_DIR):
    os.makedirs(BASE_DIR)


def printt(*args, **kwargs):
    """
    Prints with current system time
    """
    current_time = datetime.datetime.now().time()
    print(current_time.strftime("%H:%M:%S") + "   ", *args, **kwargs)


########################
# All the functions below for triadic census are taken from networkx library, 
# with the only modification of adding a tqdm wrapper to the main loop.
########################

TRICODES = (
    1,
    2,
    2,
    3,
    2,
    4,
    6,
    8,
    2,
    6,
    5,
    7,
    3,
    8,
    7,
    11,
    2,
    6,
    4,
    8,
    5,
    9,
    9,
    13,
    6,
    10,
    9,
    14,
    7,
    14,
    12,
    15,
    2,
    5,
    6,
    7,
    6,
    9,
    10,
    14,
    4,
    9,
    9,
    12,
    8,
    13,
    14,
    15,
    3,
    7,
    8,
    11,
    7,
    12,
    14,
    15,
    8,
    14,
    13,
    15,
    11,
    15,
    15,
    16,
)

#: The names of each type of triad. The order of the elements is
#: important: it corresponds to the tricodes given in :data:`TRICODES`.
TRIAD_NAMES = (
    "003",
    "012",
    "102",
    "021D",
    "021U",
    "021C",
    "111D",
    "111U",
    "030T",
    "030C",
    "201",
    "120D",
    "120U",
    "120C",
    "210",
    "300",
)


#: A dictionary mapping triad code to triad name.
TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for i, code in enumerate(TRICODES)}


def _tricode(G, v, u, w):
    """Returns the integer code of the given triad.

    This is some fancy magic that comes from Batagelj and Mrvar's paper. It
    treats each edge joining a pair of `v`, `u`, and `w` as a bit in
    the binary representation of an integer.

    """
    combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16), (w, u, 32))
    return sum(x for u, v, x in combos if v in G[u])


def my_triadic_census(G, nodelist=None):
    """Determines the triadic census of a directed graph.

    The triadic census is a count of how many of the 16 possible types of
    triads are present in a directed graph. If a list of nodes is passed, then
    only those triads are taken into account which have elements of nodelist in them.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph
    nodelist : list
        List of nodes for which you want to calculate triadic census

    Returns
    -------
    census : dict
       Dictionary with triad type as keys and number of occurrences as values.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])
    >>> triadic_census = nx.triadic_census(G)
    >>> for key, value in triadic_census.items():
    ...     print(f"{key}: {value}")
    ...
    003: 0
    012: 0
    102: 0
    021D: 0
    021U: 0
    021C: 0
    111D: 0
    111U: 0
    030T: 2
    030C: 2
    201: 0
    120D: 0
    120U: 0
    120C: 0
    210: 0
    300: 0

    Notes
    -----
    This algorithm has complexity $O(m)$ where $m$ is the number of edges in
    the graph.

    Raises
    ------
    ValueError
        If `nodelist` contains duplicate nodes or nodes not in `G`.
        If you want to ignore this you can preprocess with `set(nodelist) & G.nodes`

    See also
    --------
    triad_graph

    References
    ----------
    .. [1] Vladimir Batagelj and Andrej Mrvar, A subquadratic triad census
        algorithm for large sparse networks with small maximum degree,
        University of Ljubljana,
        http://vlado.fmf.uni-lj.si/pub/networks/doc/triads/triads.pdf

    """
    nodeset = set(G.nbunch_iter(nodelist))
    if nodelist is not None and len(nodelist) != len(nodeset):
        raise ValueError("nodelist includes duplicate nodes or nodes not in G")

    N = len(G)
    Nnot = N - len(nodeset)  # can signal special counting for subset of nodes

    # create an ordering of nodes with nodeset nodes first
    m = {n: i for i, n in enumerate(nodeset)}
    if Nnot:
        # add non-nodeset nodes later in the ordering
        not_nodeset = G.nodes - nodeset
        m.update((n, i + N) for i, n in enumerate(not_nodeset))

    # build all_neighbor dicts for easy counting
    # After Python 3.8 can leave off these keys(). Speedup also using G._pred
    # nbrs = {n: G._pred[n].keys() | G._succ[n].keys() for n in G}
    nbrs = {n: G.pred[n].keys() | G.succ[n].keys() for n in G}
    dbl_nbrs = {n: G.pred[n].keys() & G.succ[n].keys() for n in G}

    if Nnot:
        sgl_nbrs = {n: G.pred[n].keys() ^ G.succ[n].keys() for n in not_nodeset}
        # find number of edges not incident to nodes in nodeset
        sgl = sum(1 for n in not_nodeset for nbr in sgl_nbrs[n] if nbr not in nodeset)
        sgl_edges_outside = sgl // 2
        dbl = sum(1 for n in not_nodeset for nbr in dbl_nbrs[n] if nbr not in nodeset)
        dbl_edges_outside = dbl // 2

    # Initialize the count for each triad to be zero.
    census = {name: 0 for name in TRIAD_NAMES}
    # Main loop over nodes
    for v in tqdm(nodeset):
        vnbrs = nbrs[v]
        dbl_vnbrs = dbl_nbrs[v]
        if Nnot:
            # set up counts of edges attached to v.
            sgl_unbrs_bdy = sgl_unbrs_out = dbl_unbrs_bdy = dbl_unbrs_out = 0
        for u in vnbrs:
            if m[u] <= m[v]:
                continue
            unbrs = nbrs[u]
            neighbors = (vnbrs | unbrs) - {u, v}
            # Count connected triads.
            for w in neighbors:
                if m[u] < m[w] or (m[v] < m[w] < m[u] and v not in nbrs[w]):
                    code = _tricode(G, v, u, w)
                    census[TRICODE_TO_NAME[code]] += 1

            # Use a formula for dyadic triads with edge incident to v
            if u in dbl_vnbrs:
                census["102"] += N - len(neighbors) - 2
            else:
                census["012"] += N - len(neighbors) - 2

            # Count edges attached to v. Subtract later to get triads with v isolated
            # _out are (u,unbr) for unbrs outside boundary of nodeset
            # _bdy are (u,unbr) for unbrs on boundary of nodeset (get double counted)
            if Nnot and u not in nodeset:
                sgl_unbrs = sgl_nbrs[u]
                sgl_unbrs_bdy += len(sgl_unbrs & vnbrs - nodeset)
                sgl_unbrs_out += len(sgl_unbrs - vnbrs - nodeset)
                dbl_unbrs = dbl_nbrs[u]
                dbl_unbrs_bdy += len(dbl_unbrs & vnbrs - nodeset)
                dbl_unbrs_out += len(dbl_unbrs - vnbrs - nodeset)
        # if nodeset == G.nodes, skip this b/c we will find the edge later.
        if Nnot:
            # Count edges outside nodeset not connected with v (v isolated triads)
            census["012"] += sgl_edges_outside - (sgl_unbrs_out + sgl_unbrs_bdy // 2)
            census["102"] += dbl_edges_outside - (dbl_unbrs_out + dbl_unbrs_bdy // 2)

    # calculate null triads: "003"
    # null triads = total number of possible triads - all found triads
    total_triangles = (N * (N - 1) * (N - 2)) // 6
    triangles_without_nodeset = (Nnot * (Nnot - 1) * (Nnot - 2)) // 6
    total_census = total_triangles - triangles_without_nodeset
    census["003"] = total_census - sum(census.values())

    return census



if __name__ == "__main__":
    printt("Loading graph...")
    G = nx.read_graphml(osp.join(BASE_DIR, "users_clean.graphml"))

    if not os.path.isfile(BASE_DIR + "triadic_census_original.json"):
        printt("Computing triadic census for the original graph...")
        triadic_census = my_triadic_census(G)
        with open(BASE_DIR + "triadic_census_original.json", "w") as f:
            json.dump(triadic_census, f)
    else:
        printt("Found triadic census for the original graph.")
        with open(BASE_DIR + "triadic_census_original.json", "r") as f:
            triadic_census = json.load(f)

    df_G = pd.DataFrame.from_dict({k: [v] for k, v in triadic_census.items()})

    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    ntimes = 8
    null_census = []
    for i in range(ntimes):

        if not os.path.isfile(BASE_DIR + f"triadic_census_null_{i}.json"):
            printt(f"Computing triadic census for the null graph {i}...")
            G_null = nx.directed_configuration_model(in_degrees, out_degrees, create_using=nx.DiGraph)
            res = my_triadic_census(G_null)
            with open(BASE_DIR + f"triadic_census_null_{i}.json", "w") as f:
                json.dump(res, f)
        else:
            printt(f"Found triadic census for the null graph {i}.")
            with open(BASE_DIR + f"triadic_census_null_{i}.json", "r") as f:
                res = json.load(f)
        null_census.append(pd.DataFrame.from_dict({k: [v] for k, v in res.items()}))
        
    df_null = pd.concat(null_census)
    Z = (df_G.loc[0] - df_null.mean(axis=0)) / df_null.std(axis=0)

    printt("Saving results...")
    Z.to_csv(BASE_DIR + "triadic_census_zscores.csv")
    printt("Done!")
