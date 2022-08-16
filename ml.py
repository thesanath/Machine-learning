import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)   

inputs=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

training_outputs=np.array([[0,1,1,0]]).T 



synaptic_weights=np.random.random((3,1))

print('weights')

print(synaptic_weights)
for iteration in range(100000):

    input_layer =inputs

    outputs=sigmoid(np.dot(input_layer,synaptic_weights))

    error=training_outputs-outputs

    adjustments=error*sigmoid_derivative(outputs)
    
    synaptic_weights+=np.dot(input_layer.T,adjustments)


print('Outputs after training')

print(outputs)
