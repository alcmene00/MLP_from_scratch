import time
import numpy as np
from data import load_mnist
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid


def forward_propagation(x, weights, total_layers_number):
    x_1 = np.ones((1, 1))
    layers = [None] * total_layers_number
    layers[0] = x  # INPUT LAYER == IMAGES
    for i in range(total_layers_number - 1):
        layers[i] = np.append(x_1, layers[i], axis=0)
        U1 = np.dot(weights[i], layers[i])
        layers[i + 1] = sigmoid(U1)
    return layers


def back_propagation(layers, total_layers_number, labels, learning_rate, weights):
    Dw = np.empty(total_layers_number - 1, dtype=object)

    # FROM OUTPUT LAYER
    i = total_layers_number - 1
    delta_i = error(layers[i], labels) * sigmoid_derivative(layers[i])  # CALCULATE DELTA
    Dw[i - 1] = learning_rate * np.dot(delta_i, layers[i - 1].transpose())  # CALCULATE DW

    # FROM HIDDEN LAYERS
    for i in range(total_layers_number - 2, 0, -1):
        current_weight_t = weights[i].transpose()
        current_layer = layers[i]
        previous_layer = layers[i - 1]
        delta_i = np.dot(current_weight_t[1:current_weight_t.shape[0]], delta_i) * \
                  sigmoid_derivative(current_layer[1:current_layer.shape[0]])  # CALCULATE DELTA
        Dw[i - 1] = learning_rate * np.dot(delta_i, previous_layer.transpose())  # CALCULATE DW

    # UPDATE WEIGHTS
    for i in range(total_layers_number - 1):
        weights[i] += Dw[i]

    return weights


def sigmoid(x):  # NUMERICALLY- STABLE SIGMOID FUNCTION
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def sigmoid_derivative(y):
    return y * (1 - y)


def get_prediction(x):  # FIND NETWORK'S PREDICTION FROM THE OUTPUT LAYER'S MATRIX
    return np.argmax(x, 0)[0]


def error(output, labels):  # CALCULATE OUTPUT'S ERROR
    labels_array = np.zeros((10, 1))  # ONE HOT ENCODING
    labels_array[labels] = 1
    return labels_array - output


def mean_square_error(errors):
    mse = 0
    for j in range(errors.shape[0]):
        mse += errors[j][0] ** 2
    return mse/errors.shape[0]


def loss_function(errors):
    loss = 0
    for i in range(len(errors)):
        loss += mean_square_error(errors[i])
    return loss / len(errors)


def is_correct(prediction, y):
    return prediction == y


def MyKNN(neighbor, X_train, y_train, X_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=neighbor)
    classifier.fit(X_train, y_train)
    scr = classifier.score(X_test, y_test)
    print("--KNN with", neighbor, "neighbor, accuracy= ", scr)


def MyNearestCentroid(X_train, y_train, X_test, y_test):
    classifier = NearestCentroid()
    classifier.fit(X_train, y_train)
    scr = classifier.score(X_test, y_test)
    print("--Nearest centroid, accuracy= ", scr)


def MyNeuralNetwork(images_train, labels_train, images_test, labels_test):
    hidden_layers_number = 1
    learning_rate = 0.1
    epochs = 40

    total_layers_number = hidden_layers_number + 2
    layer_neurons = np.empty(total_layers_number, dtype=object)
    layer_neurons[0] = images_train.shape[1]  # INPUT LAYER NUMBER OF NEURONS
    for i in range(1, hidden_layers_number + 1):
        layer_neurons[i] = 400  # HIDDEN LAYER NUMBER OF NEURONS
    layer_neurons[hidden_layers_number + 1] = 10  # OUTPUT LAYER NUMBER OF NEURONS

    # GENERATE A RANDOM BIAS MATRIX
    bias = [None] * (total_layers_number - 1)
    for i in range(total_layers_number - 1):
        bias[i] = np.zeros((layer_neurons[i + 1], 1))  # BIAS FROM INPUT TO HIDDEN LAYER

    # GENERATE RANDOM WEIGHTS MATRIX
    weights = [None] * (total_layers_number - 1)
    for i in range(total_layers_number - 1):
        weights[i] = np.random.rand(layer_neurons[i + 1], layer_neurons[i]) - 0.5
        weights[i] = np.append(bias[i], weights[i], axis=1)

    # STOCHASTIC GRADIENT DESCENT
    for epoch in range(epochs):
        corrects = 0
        errors = [None] * images_train.shape[0]
        i = 0
        start = time.time()
        for image, label in zip(images_train, labels_train):  # TRAINING
            image = image[:, np.newaxis]
            layers = forward_propagation(image, weights, total_layers_number)
            errors[i] = error(layers[total_layers_number - 1], label)
            if is_correct(get_prediction(layers[total_layers_number - 1]), label):
                corrects = corrects + 1
            weights = back_propagation(layers, total_layers_number, label, learning_rate, weights)
            i = i + 1
        end = time.time()
        train_time = end - start
        print("EPOCH: " + str(epoch + 1))
        print(
            "Train Accuracy: " + f"{round((corrects / images_train.shape[0]) * 100, 2)}% " +
            "Loss: " + f"{round(loss_function(errors), 3)}" + " Time: " + f"{round(train_time, 2)} sec.")
        corrects_test = 0
        errors = [None] * images_test.shape[0]
        i = 0
        start = time.time()
        for image, label in zip(images_test, labels_test):  # TESTING
            image = image[:, np.newaxis]
            layers = forward_propagation(image, weights, total_layers_number)
            errors[i] = error(layers[total_layers_number - 1], label)
            if is_correct(get_prediction(layers[total_layers_number - 1]), label):
                corrects_test = corrects_test + 1
            i = i + 1
        end = time.time()
        test_time = end - start
        print("Test Accuracy: " + f"{round((corrects_test / images_test.shape[0]) * 100, 2)}%" +
              " Loss: " + f"{round(loss_function(errors), 3)}" + " Time: " + f"{round(test_time, 2)} sec.\n")


def main():
    images_train, labels_train, images_test, labels_test = load_mnist()

    MyNeuralNetwork(images_train, labels_train, images_test, labels_test)
    # MyKNN(1, images_train, labels_train, images_test, labels_test)
    # MyKNN(3, images_train, labels_train, images_test, labels_test)
    # MyNearestCentroid(images_train, labels_train, images_test, labels_test)


main()
