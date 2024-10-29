import numpy as np
import pandas as pd
from graphviz import Digraph


class Variable:
    def __init__(self, value, grad=0.0, _prev=(), _op=''):
        self.value = value  # The actual value of the variable
        self.grad = grad  # Gradient of the variable (initialized to 0)
        self._prev = set(_prev)  # Previous variables (inputs) used to create this variable
        self._op = _op  # The operation that produced this variable (for visualization)
        self._backward = lambda: None  # Function to compute the gradient for this variable

    def __add__(self, other):
        # Create a new Variable for the sum
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value, _prev=(self, other), _op='+')

        # Define the backward pass for addition
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # Create a new Variable for the product
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value * other.value, _prev=(self, other), _op='*')

        # Define the backward pass for multiplication
        # The backward function performs backpropagation by computing gradients for each variable in the computational graph
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backward = _backward
        return out

    # The backward pass defines how gradients are calculated for each operation using the chain rule.
    def backward(self):
        # Set the gradient of the output variable to 1
        self.grad = 1.0
        # Perform a topological sort to ensure correct gradient computation order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Traverse backward in reverse topological order
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad})"
    # taking input
x = Variable(5.0)
y = (x + 2) * 3
# Perform backpropagation
y.backward()
print("Value of y:", y)
print("Gradient of x:", x)

# drawing graph
def draw_graph(var):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})

    # Recursive function to add nodes and edges to the graph
    def add_nodes_edges(v):
        if v not in seen:
            # Add the node for the variable
            seen.add(v)
            node_id = str(id(v))
            dot.node(node_id, f"{v._op} | {v.value:.4f} | grad={v.grad:.4f}", shape='record')

            # Add edges for each previous variable
            for child in v._prev:
                child_id = str(id(child))
                dot.edge(child_id, node_id)
                add_nodes_edges(child)

    seen = set()
    add_nodes_edges(var)

    return dot


# Draw the graph for y
dot = draw_graph(y)
dot.render('computational_graph', view=True)