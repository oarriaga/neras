import numpy as np
import matplotlib.pyplot as plt


class Dense():
    def __init__(self, output_dim=None, input_dim=None,
                 activation='linear'):

        self.input_shape = input_dim
        self.output_shape = output_dim
        self.weights = (None, None)
        self.bias = None
        self.name = 'dense'
        self.activation = activation
        self.deltas = None

    def initialize_weights(self):
        self.weights = np.random.rand(
            self.output_shape, self.input_shape)
        self.bias = np.random.rand(self.output_shape, 1)

    def forward_pass(self, training_input):
        net = np.dot(self.weights, training_input) + self.bias
        y_predicted = self.gamma(net)
        return y_predicted

    def gamma(self, net):
        if self.activation == 'linear':
            return net
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-1.0 * net))

    def gamma_derivative(self, net):
        if self.activation == 'linear':
            return np.ones_like(net)
        if self.activation == 'sigmoid':
            return net * (1 - net)


class Sequential():
    def __init__(self):
        self.layers = []
        self.output_shapes = []
        self.loss = None
        self.optimizer = None
        self.learning_rate = None
        self.compiled = False

    def add(self, layer):
        if (len(self.layers) == 0 and layer.input_shape is None):
            raise Exception('First layer should have input_shape')
        if (len(self.layers) != 0 and layer.input_shape is not None):
            raise Exception(" 'input_shape' only allowed in first layer")

        layer.name = layer.name + '_' + str(len(self.layers))
        if len(self.layers) > 0:
            layer.input_shape = self.layers[-1].output_shape
        layer.initialize_weights()
        self.layers.append(layer)

    def compile(self, optimizer='SGD', learning_rate=0.01, loss='MSE'):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.compiled = True

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def fit(self, X, y, num_epochs, verbose=True):
        if not self.compiled:
            raise Exception("Model was not 'compiled'")
        loss_history = []
        for epoch in range(num_epochs):
            self.update_weights(X, y)
            total_loss = np.sum(self.calculate_loss(X, y))
            loss_history.append(total_loss)
            if verbose:
                print('epoch:', epoch / num_epochs, 'loss:', total_loss)
        return loss_history

    def calculate_loss(self, X, y):
        if self.loss == 'MSE':
            squared_error_sum = 0.0
            for sample_arg in range(len(X)):
                x_sample = X[sample_arg:sample_arg + 1].T
                y_sample = y[sample_arg:sample_arg + 1].T
                y_predicted = self.predict(x_sample)
                squared_error = np.square(y_predicted - y_sample)
                squared_error_sum = squared_error_sum + squared_error
            return squared_error_sum / len(X)

    def update_weights(self, X, y):
        for sample_arg in range(len(X)):
            X_sample = X[sample_arg:(sample_arg + 1)].T
            y_sample = y[sample_arg:(sample_arg + 1)].T
            layer_outputs, layer_input = [], X_sample
            """
            for layer_arg, layer in enumerate(self.layers):
                print(layer_arg)
                print('layer_input', layer_input.shape)
                if layer_arg > 0:
                    layer_input = layer_outputs[-1]
                    print('layer_output_shape', layer_outputs[-1].shape)
                layer_output = layer.forward_pass(layer_input)
                print(layer.name)
                print('layer_output', layer_output.shape)
                layer_outputs.append(layer_output)
            """

            layer_outputs = []
            tensor = X_sample
            for layer in self.layers:
                tensor = layer.forward_pass(tensor)
                layer_outputs.append(tensor)

            # print('layer_outputs_final length:', len(layer_outputs))
            # for layer_o in layer_outputs:
            #     print('layer_o', layer_o.shape)
            layer_args = list(range(len(self.layers))[::-1])
            for layer_arg, layer in zip(layer_args, reversed(self.layers)):
                layer_output = layer_outputs[layer_arg]
                if layer_arg == (len(self.layers) - 1):
                    loss_derivative = y_sample - layer_output
                    gamma_derivative = layer.gamma_derivative(layer_output)
                    layer.deltas = gamma_derivative * loss_derivative
                    # print('loss_derivative', loss_derivative.shape)
                    # print('gamma_derivative', gamma_derivative.shape)
                else:
                    upper_deltas = self.layers[layer_arg + 1].deltas
                    weights = self.layers[layer_arg + 1].weights
                    # print('weights_shape', weights.shape)
                    # print('upper_deltas', upper_deltas.shape)
                    weighted_error = np.dot(weights.T, upper_deltas)
                    gamma_derivative = layer.gamma_derivative(layer_output)
                    weighted_error = np.dot(weights.T, upper_deltas)
                    layer.deltas = weighted_error * gamma_derivative
                if layer_arg == 0:
                    layer_input = X_sample
                else:
                    layer_input = layer_outputs[layer_arg - 1]
                delta_weights = np.dot(layer_input, layer.deltas.T).T
                layer.weights = layer.weights + (self.learning_rate * delta_weights)


def create_xor_data():
    x1, y1 = np.array([[1, 0]]), np.array([[1]])
    x2, y2 = np.array([[0, 1]]), np.array([[1]])
    x3, y3 = np.array([[1, 1]]), np.array([[0]])
    x4, y4 = np.array([[0, 0]]), np.array([[0]])
    x = np.vstack((x1, x2, x3, x4))
    y = np.vstack((y1, y2, y3, y4))
    return x, y

X, y = create_xor_data()


model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='MSE', learning_rate=0.01)
loss_history = model.fit(X, y, num_epochs=1000000, verbose=True)

plt.plot(loss_history)
plt.title('Loss history with random weights for XOR')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.show()

x_test = np.array([[1, 0]])
print('Prediction of [1, 0]:', model.predict(x_test.T))
x_test = np.array([[0, 1]])
print('Prediction of [0, 1]:', model.predict(x_test.T))
x_test = np.array([[1, 1]])
print('Prediction of [1, 1]:', model.predict(x_test.T))
x_test = np.array([[0, 0]])
print('Prediction of [0, 0]:', model.predict(x_test.T))

print(X)
print(y)

x_data = np.arange(-2, 2, .1)
y_data = np.arange(-2, 2, .1)
for x in x_data:
    for y in y_data:
        input_point = np.array([[x, y]])
        class_point = model.predict(input_point.T)
        class_point = class_point > .5
        if class_point[0][0]:
            col = 'ro'
        else:
            col = 'bo'
        plt.plot(x, y, col, markersize=5)

plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('Decision boundary for the XOR')
plt.show()
