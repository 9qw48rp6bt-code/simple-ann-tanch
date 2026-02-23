import random
import math

# -----------------------------
# Settings
# -----------------------------
random.seed(42)  # optional: makes random results reproducible

# Inputs
x1 = 0.05
x2 = 0.10

# Bias values
b1 = 0.5  # hidden layer bias
b2 = 0.7  # output layer bias

# -----------------------------
# Helper functions
# -----------------------------
def rand_w():
    """Random weight in [-0.5, 0.5]."""
    return random.uniform(-0.5, 0.5)

def tanh(z):
    return math.tanh(z)

# -----------------------------
# Random weights
# Network: 2-2-2
# Input -> Hidden: w1..w4
# Hidden -> Output: w5..w8
# -----------------------------
w1 = rand_w()  # x1 -> h1
w2 = rand_w()  # x2 -> h1
w3 = rand_w()  # x1 -> h2
w4 = rand_w()  # x2 -> h2

w5 = rand_w()  # h1 -> o1
w6 = rand_w()  # h2 -> o1
w7 = rand_w()  # h1 -> o2
w8 = rand_w()  # h2 -> o2

# -----------------------------
# Forward pass
# -----------------------------
net_h1 = x1 * w1 + x2 * w2 + b1
net_h2 = x1 * w3 + x2 * w4 + b1

out_h1 = tanh(net_h1)
out_h2 = tanh(net_h2)

net_o1 = out_h1 * w5 + out_h2 * w6 + b2
net_o2 = out_h1 * w7 + out_h2 * w8 + b2

out_o1 = tanh(net_o1)
out_o2 = tanh(net_o2)

# -----------------------------
# Print results
# -----------------------------
print("=== Random Weights (in [-0.5, 0.5]) ===")
print(f"w1={w1:.6f}, w2={w2:.6f}, w3={w3:.6f}, w4={w4:.6f}")
print(f"w5={w5:.6f}, w6={w6:.6f}, w7={w7:.6f}, w8={w8:.6f}")

print("\n=== Forward Pass with tanh ===")
print(f"Inputs: x1={x1}, x2={x2}")
print(f"Hidden net: net_h1={net_h1:.6f}, net_h2={net_h2:.6f}")
print(f"Hidden out: out_h1={out_h1:.6f}, out_h2={out_h2:.6f}")
print(f"Output net: net_o1={net_o1:.6f}, net_o2={net_o2:.6f}")
print(f"Network Output: o1={out_o1:.6f}, o2={out_o2:.6f}")
