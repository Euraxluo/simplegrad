# -*- coding: utf-8 -*- 
# Time: 2022-01-21 12:11
# Copyright (c) 2022
# author: Euraxluo

from typing import *
import os
import contextlib


class Profile(object):
    def __init__(self, graph: False):
        self.ctx = None
        self.name = None
        self.x = None
        self.output: List = []
        self.backward = None
        self.i = None
        self.o = None
        self.edges_node_ids = []
        if graph:
            import networkx as nx
            self.graph: nx.DiGraph = nx.DiGraph()
        else:
            self.graph = None

    @contextlib.contextmanager
    def log(self, ctx, name, x, backward=False):
        self.ctx, self.name, self.x, self.output, self.backward = ctx, f"back_{name}" if backward else name, x, [], backward
        yield self
        if self.graph is not None:
            for x in self.x:
                for y in self.output:
                    self.graph.add_edge(id(x.data), id(y.data), label=self.name, color="blue" if self.backward else "black")
                    self.graph.nodes[id(x.data)]['label'], self.graph.nodes[id(y.data)]['label'] = str(x.shape), str(y.shape)
                    self.edges_node_ids.append(id(x.data))
                    self.edges_node_ids.append(id(y.data))
                    if self.i is None:
                        self.i = id(x.data)
                    self.o = id(y.data)
            if self.backward:
                for s in self.ctx.saved_tensors:
                    for o in self.output:
                        if id(s) not in self.edges_node_ids:
                            self.graph.add_edge(id(s), id(o.data), label=self.name, color="red")
                            self.graph.nodes[id(s)]['label'] = f"<{str(s)}>"
            else:
                for x in self.x:
                    for s in self.ctx.saved_tensors:
                        if id(x.data) != id(s) and id(s) not in self.edges_node_ids:
                            self.graph.add_edge(id(s), id(x.data), label=self.name, color="purple")
                            self.graph.nodes[id(s)]['label'] = f"<{str(s)}>"

    def show(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        edge_color = []
        for i in self.graph.edges():
            edge_color.append(self.graph.edges[i]['color'])
            del self.graph.edges[i]['color']

        node_labels = {}
        for k, v in self.graph.nodes().items():
            node_labels[k] = f"{k}\n" \
                             f"{v.get('label', '')}\n" \
                             f"{'input' if k == self.i else 'output' if self.o == k else ''}\n"

        pos = nx.circular_layout(self.graph)
        nx.draw(self.graph, pos=pos,
                edge_color=edge_color)
        nx.draw_networkx_edge_labels(self.graph, pos=pos, alpha=0.5, font_size=7)
        nx.draw_networkx_labels(self.graph, pos=pos, labels=node_labels, font_size=8, alpha=0.5)
        plt.show()

    def save(self, graph_path: str = '/tmp/net.graph.dot'):
        ...


GRAPH = os.getenv("GRAPH", None) is not None
profile = Profile(graph=GRAPH)

__all__ = ["profile"]
