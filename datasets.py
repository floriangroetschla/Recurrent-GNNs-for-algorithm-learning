import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx
from scipy.spatial import Delaunay


class PrefixSumK():
    # Creates a PrefixSum on a path - output must be sum mod 2
    # input is 2 values [value, isRoot]
    def __init__(self, k = 2, inp = 2):
        super().__init__()
        self.num_classes = k
        self.num_features = 2
        self.name = "PrefixSum mod K"
        self.inp = inp

    def gen_graph(self, s):
        n = len(s)
        rand_perm = np.arange(n)
        G = nx.path_graph(rand_perm)
        leafs = np.random.permutation([x for x in G.nodes() if G.degree(x)==1])

        root = rand_perm[0]
        labels = [[0.0, 0.0] for i in range(n)]
        ylabels = [[0.0] for i in range(n)]
        labels[root] = [0.0, 1.0]
        
        counter = 0
        for i,node in enumerate(rand_perm):        
            x = int(s[i])
            labels[node][0] = s[i]
            counter = (counter+ x)%self.num_classes
            ylabels[node] = counter

        dG = from_networkx(G)
        dG.y = torch.tensor(ylabels)
        dG.x = torch.tensor(labels)

        return dG

    def makedata(self, num_graphs = 200, num_nodes = 8, allow_sizes = False):
        binary_strs = []
        while len(binary_strs) < num_graphs:
            graph_size = num_nodes
            if allow_sizes:
                graph_size = np.random.randint(2, graph_size+1)
            ss = [np.random.randint(0,self.inp)*1.0 for _ in range(graph_size)]
            if ss not in binary_strs:
                binary_strs.append(ss)
        return [self.gen_graph(s) for s in binary_strs]

class PrefixSum():
    # Creates a PrefixSum on a path - output must be sum mod 2
    # input is one-hot [value, isRoot]
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.num_features = 4
        self.name = "PrefixSum"

    def gen_graph(self, s):
        n = len(s)
        rand_perm = np.arange(n)
        G = nx.path_graph(rand_perm)
        leafs = np.random.permutation([x for x in G.nodes() if G.degree(x)==1])

        root = rand_perm[0]
        labels = [[0.0, 0.0, 1.0, 0.0] for i in range(n)]
        ylabels = [[0.0] for i in range(n)]
        labels[root] = [0.0, 0.0, 0.0, 1.0]
        
        counter = 0
        for i,node in enumerate(rand_perm):        
            x = int(s[i])
            labels[node][x] = 1.0
            counter = (counter+ x)%2
            ylabels[node] = counter

        dG = from_networkx(G)
        dG.y = torch.tensor(ylabels)
        dG.x = torch.tensor(labels)

        dG.edge_attr = torch.ones(G.number_of_edges()*2, 1)

        return dG

    def makedata(self, num_graphs = 200, num_nodes = 8, allow_sizes = False):
        binary_strs = []
        while len(binary_strs) < num_graphs:
            graph_size = num_nodes
            if allow_sizes:
                graph_size = np.random.randint(2, graph_size+1)
            ss = ''.join([str(np.random.randint(0,2)) for _ in range(graph_size)])
            if ss not in binary_strs:
                binary_strs.append(ss)
        return [self.gen_graph(s) for s in binary_strs]

class Trees():
    # Creates a Tree and marks the shortest path between two nodes
    # input is one-hot [isEndpoint]
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.num_features = 2
        self.name = "ShortestPathTrees"

    def gen_graph(self, num_nodes, num):
        nx_graph = nx.random_tree(n=num_nodes, seed=num)
        tree = from_networkx(nx_graph)
        tree.x = torch.zeros(num_nodes, 2)
        tree.y = torch.zeros(num_nodes)
        tree.x[0][1] = 1
        tree.x[1][1] = 1
        shortest_path = nx.shortest_path(nx_graph, source=0, target=1)
        for node in shortest_path:
            tree.y[node] = 1
        for node in range(num_nodes):
            if tree.x[node][1] == 0:
                tree.x[node][0] = 1
        tree.edge_attr = torch.ones(nx_graph.number_of_edges()*2, 1)
        return tree

    def makedata(self, num_graphs = 200, num_nodes = 8, allow_sizes = False):
        return [self.gen_graph(num_nodes, i) for i in range(num_graphs)]

class MidPoint():
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.num_features = 2
        self.name = "MidPoint"

    def gen_graph(self, graph_size, posA, posB):
        n = graph_size
        rand_perm = np.arange(n)
        G = nx.path_graph(rand_perm)
        leafs = np.random.permutation([x for x in G.nodes() if G.degree(x)==1])

        labels = [[1.0, 0.0] for i in range(n)]
        ylabels = [0.0 for i in range(n)]
        
        labels[posA] = [0.0, 1.0]
        labels[posB] = [0.0, 1.0]
        ylabels[int((posA+posB)/2)] = 1.0


        dG = from_networkx(G)
        dG.y = torch.tensor(ylabels)
        dG.x = torch.tensor(labels)
        return dG

    def makedata(self, num_graphs = 200, num_nodes = 8, allow_sizes = False):
        graph_str = []
        graph_list = []
        while len(graph_str) < num_graphs:
            graph_size = num_nodes
            if allow_sizes:
                graph_size = np.random.randint(graph_size - 5, graph_size+1)
            posA, posB = 0,0
            for j in range(1000):
                posA = np.random.randint(0, graph_size)
                posB = np.random.randint(0, graph_size)
                if 0 <= posA and posA < graph_size and posA < posB and posB < graph_size and (posB - posA)%2 == 0:
                    success = True
                    break
            ss = f"{graph_size},{posA},{posB}"
            if not success or ss in graph_str:
                continue
            graph_str.append(ss)
            graph_list.append(self.gen_graph(graph_size, posA, posB))
        return graph_list

class Cycles():
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.num_features = 4
        self.name = "Cycles"

    def gen_graph(self, graph_size, posA, posB):
        n = graph_size
        rand_perm = np.arange(n)
        G = nx.path_graph(rand_perm)

        leafs = np.random.permutation([x for x in G.nodes() if G.degree(x)==1])
        G.add_edge(leafs[0], leafs[1])

        labels = [[1.0, 0.0] for i in range(n)]
        ylabels = [0.0 for i in range(n)]
        
        labels[posA] = [0.0, 1.0]
        labels[posB] = [0.0, 1.0]

        dist = posB - posA
        odist = (graph_size - dist)//2
        ylabels[(posA + posB)//2] = 1.0
        ylabels[(posA-odist+graph_size)%graph_size] = 1.0

        dG = from_networkx(G)
        dG.y = torch.tensor(ylabels)
        dG.x = torch.tensor(labels)
        return dG

    def makedata(self, num_graphs=10, num_nodes=8, allow_sizes=False):
        graph_str = []
        graph_list = []
        while len(graph_str) < num_graphs:
            graph_size = num_nodes
            if allow_sizes:
                graph_size = np.random.randint(4, graph_size+1)
            posA, posB = 0,0
            succ = False
            for i in range(100):
                posA = 0
                posB = posA + np.random.randint(1, graph_size-posA)
                if 0 <= posA and posA < graph_size and posA < posB and posB < graph_size:
                    if posB - posA <= graph_size//2 and (posB - posA)%2 == 0:
                        succ = True
                        break

            ss = f"{graph_size},{posB-posA}"
            if not succ or ss in graph_str:
                continue

            graph_str.append(ss)
            graph_list.append(self.gen_graph(graph_size, posA, posB))

        return graph_list

def randomgraph(n, **args):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    tree = set()
    nodes = list(range(n))
    current = np.random.choice(nodes)
    tree.add(current)
    while(len(tree) < n):
        nxt = np.random.choice(nodes)
        if not nxt in tree:
            tree.add(nxt)
            g.add_edge(current, nxt)
            g.add_edge(nxt, current)
        current = nxt
    for _ in range(n//5):
        i, j = np.random.permutation(n)[:2]
        while g.has_edge(i,j):
            i, j = np.random.permutation(n)[:2]
        g.add_edge(i, j)
        g.add_edge(j, i)
    return g

def get_localized_distances(g, n):
    seen = set()
    distances = {}
    queue = [(n, 0)]
    while queue:
        node, distance = queue.pop(0)
        if node in distances and distances[node] < distance:
            continue
        distances[node] = distance
        for nb in g.neighbors(node):
            if nb not in seen:
                seen.add(node)
                queue.append((nb, distance + 1))
    return [distances[i] for i in range(g.number_of_nodes())]

class Distance():
    def __init__(self, num_graphs=200, num_nodes=12):
        super().__init__()
        self.num_features = 2
        self.num_classes = 2
        self.name = "Distance"

    def gen_graph(self, num_nodes):
        g = randomgraph(num_nodes)

        origin = np.random.randint(0, num_nodes)
        queue = [(origin, 0)]
        seen = {origin}
        even = set()

        while queue:
            node, distance = queue.pop(0)
            if distance % 2 == 0:
                even.add(node)
            for nb in g.neighbors(node):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, distance + 1))
        data = from_networkx(g)
        data.x = torch.tensor([[1.0,0.0] if x != origin else [0.0,1.0] for x in range(num_nodes)])
        #data.x[origin:origin+1,:] = torch.ones(1, self.num_features).float()
        #data.x[origin] = torch.ones([0.0,1.0])

        distances = get_localized_distances(g, origin)
        #data.diameter = max(distances)
        #data.distances = torch.tensor(distances).unsqueeze(1)
        data.edge_attr = torch.ones(g.number_of_edges()*2, 1)

        data.y = torch.tensor([0.0 if n in even else 1.0 for n in range(num_nodes)])

        return data

    def makedata(self, num_graphs = 200, num_nodes = 8, allow_sizes = False):
        return [self.gen_graph(num_nodes) for _ in range(num_graphs)]


class DistanceK():
    def __init__(self, k = 2):
        super().__init__()
        self.num_features = 2
        self.num_classes = k
        self.name = "Distance"

    def gen_graph(self, num_nodes):
        g = randomgraph(num_nodes)

        origin = np.random.randint(0, num_nodes)
        queue = [(origin, 0)]
        seen = {origin}
        even = set()

        while queue:
            node, distance = queue.pop(0)
            if distance % 2 == 0:
                even.add(node)
            for nb in g.neighbors(node):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, distance + 1))
        data = from_networkx(g)
        data.x = torch.tensor([[1.0,0.0] if x != origin else [0.0,1.0] for x in range(num_nodes)])
        #data.x[origin:origin+1,:] = torch.ones(1, self.num_features).float()
        #data.x[origin] = torch.ones([0.0,1.0])

        #distances = get_localized_distances(g, origin)
        distances = nx.shortest_path_length(g, origin)
        #data.diameter = max(distances)
        #data.distances = torch.tensor(distances).unsqueeze(1)
        data.edge_attr = torch.ones(g.number_of_edges()*2, 1)

        data.y = torch.tensor([distances[n]%self.num_classes for n in range(num_nodes)])
        #print(torch.tensor([0.0 if n in even else 1.0 for n in range(num_nodes)]))

        return data

    def makedata(self, num_graphs = 200, num_nodes = 8, allow_sizes = False):
        return [self.gen_graph(num_nodes) for _ in range(num_graphs)]

class Distance_Delaunay():
    def __init__(self, num_graphs=200, num_nodes=12):
        super().__init__()
        self.num_features = 2
        self.num_classes = 2
        self.name = "Distance_Delaunay"

    def gen_graph(self, num_nodes):
        points = np.random.rand(num_nodes, 2)

        triangulation = Delaunay(points).vertex_neighbor_vertices
        G = nx.Graph()

        for i in range(num_nodes):
            G.add_node(i)

        for i in range(len(triangulation[0])-1):
            for j in range(triangulation[0][i], triangulation[0][i+1]):
                G.add_edge(i, triangulation[1][j])

        origin = np.random.randint(0, num_nodes)

        data = from_networkx(G)
        data.x = torch.tensor([[1.0,0.0] if x != origin else [0.0,1.0] for x in range(num_nodes)])

        distances = nx.single_source_shortest_path_length(G, origin)

        data.y = torch.tensor([distances[n]%2 for n in range(num_nodes)])
        data.edge_attr = torch.ones(G.number_of_edges()*2, 1)

        return data

    def makedata(self, num_graphs = 200, num_nodes = 8, allow_sizes = False):
        return [self.gen_graph(num_nodes) for _ in range(num_graphs)]


def check_solvability(dataset):
    seen_hashes_per_label = {}
    solvable = True
    for graph in dataset:
        graph_nx = to_networkx(graph, to_undirected=True, node_attrs=['x', 'y'])
        graph_hash = nx.weisfeiler_lehman_subgraph_hashes(graph_nx, node_attr='x', iterations=graph_nx.number_of_nodes())
        for node in graph_nx.nodes:
            hash = graph_hash[node][-1]
            node_label = graph_nx.nodes[node]['y']
            if node_label in seen_hashes_per_label:
                seen_hashes_per_label[node_label].add(hash)
            else:
                seen_hashes_per_label[node_label] = set()
                seen_hashes_per_label[node_label].add(hash)

    seen_hashes = set()
    unique_labels = seen_hashes_per_label.keys()
    for label in unique_labels:
        if not solvable:
            break
        for hash in seen_hashes_per_label[label]:
            if hash in seen_hashes:
                solvable = False
                break
            else:
                seen_hashes.add(hash)
    return solvable



if __name__ == '__main__':
    print("Welcome to DATA.PY -- generate some graphs?")

    datasets = [PrefixSumK(5), PrefixSum(), Trees(), Distance(), DistanceK(5), Distance_Delaunay()]

    for dataset in datasets:
        print(dataset.name)
        data = dataset.makedata(num_graphs=100, num_nodes=10)
        solvable = check_solvability(data)
        if solvable:
            print('All good')
        else:
            print('Dataset not solvable!!!')

    exit()
