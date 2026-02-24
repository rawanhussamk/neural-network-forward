import numpy as np 

input = np.array([0.6, -0.3])

hidden_layer_Weight = np.random.uniform(-0.5, 0.5, (2, 2))
output_layer_Weight = np.random.uniform(-0.5, 0.5, (2, 2))

bias1 = 0.5
bias2 = 0.7

h1net = np.dot(hidden_layer_Weight, input) + bias1
h1output = np.tanh(h1net)

output_net = np.dot(output_layer_Weight, h1output) + bias2
output = np.tanh(output_net)

print("Hidden layer output:", h1output)
print("output:", output)