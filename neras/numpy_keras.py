import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Dense():
    def __init__(self, output_dim=None, input_dim=None,
                 activation='linear', kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):

        self.input_shape = input_dim
        self.output_shape = output_dim
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.weights = (None, None)
        self.bias = None
        self.name = 'dense'
        self.deltas = None

    def initialize_weights(self):
        if self.kernel_initializer == 'glorot_uniform':
            limit = 1 / np.sqrt(self.input_shape)
            self.weights = np.random.uniform(
                -limit, limit, (self.output_shape, self.input_shape))
        if self.kernel_initializer == 'uniform':
            self.weights = np.random.rand(self.output_shape, self.input_shape)

        if self.bias_initializer == 'zeros':
            self.bias = np.zeros((self.output_shape, 1))

        if self.bias_initializer == 'uniform':
            self.bias = np.random.rand(self.output_shape, 1)

    def forward_pass(self, X):
        return self.gamma(np.dot(self.weights, X) + self.bias)

    def gamma(self, x):
        if self.activation == 'linear':
            return x

        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-1.0 * x))

        if self.activation == 'relu':
            return np.where(x > 0, x, 0)

        if self.activation == 'tanh':
            return 2 / (1 + np.exp(-2 * x)) - 1

    def gamma_derivative(self, x):
        if self.activation == 'linear':
            return np.ones_like(x)

        if self.activation == 'sigmoid':
            return self.gamma(x) * (1 - self.gamma(x))

        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)

        if self.activation == 'tanh':
            return 1 - np.power(self.gamma(x), 2)


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

            layer_outputs = []
            tensor = X_sample
            for layer in self.layers:
                tensor = layer.forward_pass(tensor)
                layer_outputs.append(tensor)

            layer_args = list(range(len(self.layers))[::-1])
            for layer_arg, layer in zip(layer_args, reversed(self.layers)):
                layer_output = layer_outputs[layer_arg]
                if layer_arg == (len(self.layers) - 1):
                    loss_derivative = -(y_sample - layer_output)
                    gamma_derivative = layer.gamma_derivative(layer_output)
                    layer.deltas = gamma_derivative * loss_derivative
                else:
                    upper_deltas = self.layers[layer_arg + 1].deltas
                    weights = self.layers[layer_arg + 1].weights
                    weighted_error = np.dot(weights.T, upper_deltas)
                    gamma_derivative = layer.gamma_derivative(layer_output)
                    weighted_error = np.dot(weights.T, upper_deltas)
                    layer.deltas = weighted_error * gamma_derivative
                if layer_arg == 0:
                    layer_input = X_sample
                else:
                    layer_input = layer_outputs[layer_arg - 1]

                delta_weights = np.dot(layer_input, layer.deltas.T).T
                layer.weights = (
                    layer.weights - (self.learning_rate * delta_weights))

                bias_derivative = np.sum(layer.deltas, axis=0, keepdims=True)
                layer.bias = (
                    layer.bias - (self.learning_rate * bias_derivative))


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
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='MSE', learning_rate=0.01)
loss_history = model.fit(X, y, num_epochs=20000, verbose=True)

plt.plot(loss_history)
plt.title('Loss history with random weights for XOR')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.show()

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap=cm_bright, edgecolors='k')

xx, yy = np.meshgrid(np.arange(-.5, 1.5, .02), np.arange(-.5, 1.5, .02))
input_points = np.c_[xx.ravel(), yy.ravel()]
Z = []
for input_point in input_points:
    input_point = np.expand_dims(input_point, 0)
    prediction = model.predict(input_point.T)
    prediction = np.squeeze(prediction)
    Z.append(prediction)
Z = np.asarray(Z)

# Put the result into a color plot
cm = plt.cm.RdBu
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.title('Decision boundary for the XOR')
plt.xlim([-.5, 1.5])
plt.ylim([-.5, 1.5])
plt.show()
