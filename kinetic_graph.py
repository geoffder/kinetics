import numpy as np
import matplotlib.pyplot as plt

import graph_builds as gbs
from diffusion import disc2D


class Node:
    def __init__(self, name, v0):
        self.name = name
        self.v0 = v0  # initial state
        self.reset_state()
        self.out_edges = {}

    def reset_state(self):
        self.v = self.v0
        self.rec = []

    def connect(self, node, rate, agonist_sens):
        """Form an outward edge with another node (passed by reference). Value
        will transfer from this node along the edge at the given rate.
        Sensitivity of this rate to applied agonist is set by agonist_sens."""
        self.out_edges[node.name] = {
            "Node": node, "Rate": rate, "Sensitivity": agonist_sens
        }

    def update(self, delta):
        self.v += delta

    def clamp(self):
        """Prevent state value from dropping below zero or increasing above one.
        This node's value represents a proportion of the total value residing on
        the graph it belongs to."""
        self.v = min(1, max(0, self.v))

    @staticmethod
    def calc_delta(edge, agonist, v, dt):
        if edge["Sensitivity"] > 0:
            return edge["Rate"] * agonist * edge["Sensitivity"] * dt * v
        else:
            return edge["Rate"] * dt * v

    def commit_edges(self, agonist, dt):
        """Calculate outward movement of value to connected nodes and store
        results. The push method will be used by the graph to apply all deltas
        after they have all been calculated."""
        self.deltas = [
            self.calc_delta(e, agonist, self.v, dt)
            for e in self.out_edges.values()
        ]

    def push(self, edge, delta):
        edge["Node"].update(delta)
        self.update(-delta)

    def push_edges(self):
        """Apply committed deltas to connected nodes and update the local state
        value accordingly."""
        list(map(self.push, self.out_edges.values(), self.deltas))
        self.deltas = []

    def record(self):
        self.rec.append(self.v)


class KineticGraph:
    def __init__(self, dt=.01, tstop=100, name=""):
        self.dt = dt
        self.tstop = tstop
        self.name = name
        self.time = np.arange(1, tstop // dt + 1) * dt
        self.nodes = {}

    def pulse_agonist(self, pulses):
        """Takes list of triples (onset, duration, concentration) of agonist
        pulses. Build ndarray representing agonist concentration over time."""
        self.agonist = np.zeros_like(self.time)
        for (on, dur, conc) in pulses:
            self.agonist[(self.time >= on) * (self.time < (on + dur))] += conc

    def diffusion_agonist(self, vector_func):
        self.agonist = vector_func(self.time / 1000)

    def add_node(self, name, v0=0.):
        """Create node with initial state value (proportion of population)."""
        self.nodes[name] = Node(name, v0)

    def new_edge(self, n1, n2, rate, agonist_sens=0):
        """Connect two existing nodes (/w provided names) with a transition
        rate. Optionally, the connection can be externally dependent on
        agonist concentration (degree to which set by agonist_sens)."""
        self.nodes[n1].connect(self.nodes[n2], rate, agonist_sens)

    def update_edge(self, n1, n2, rate):
        """Update transfer rate from node n1 to n2 (ID'd by name property) to
        the given value."""
        if n1 in self.nodes and n2 in self.nodes[n1].out_edges:
            self.nodes[n1].out_edges[n2]["Rate"] = rate
        else:
            print("Can't update edge that does not exist.")

    def step(self, agonist_concentration):
        """Calculate values to move along each edge in graph, push the changes,
        then clamp all node values between 0 and 1."""
        for n in self.nodes.values():
            n.commit_edges(agonist_concentration, self.dt)
        for n in self.nodes.values():
            n.push_edges()
        for n in self.nodes.values():
            n.clamp()
            n.record()

    def run(self):
        """Reset state, then run steps (at dt) through until tstop."""
        for n in self.nodes.values():
            n.reset_state()

        for concentration in self.agonist:
            self.step(concentration)

        return {k: np.array(n.rec) for k, n in self.nodes.items()}


if __name__ == "__main__":
    base_pth = "/mnt/Data/NEURONoutput"
