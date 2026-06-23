from pyvis.network import Network
import networkx as nx, json

with open("data/arxiv_graph.json") as f:
    G = nx.node_link_graph(json.load(f))

net = Network(height="800px", width="100%", notebook=False)
net.from_nx(G)
net.write_html("graph.html")
print("Saved to graph.html")
