# This is a SimplePLA. For testing Python Environment Purpose
import numpy
import matplotlib.pyplot as plt


# SimplePLA Class - A over engineered implementation of Perceptron Learning Algorithm
class SimplePLA(object):
    # Constructor - with default threshold 200 and learning rate 0.01
    def __init__(self, number_of_inputs: int, threshold=100, learning_rate=0.01):
        # settings
        self.number_of_inputs = number_of_inputs
        self.threshold = threshold
        self.learning_rate = learning_rate
        # weights
        self.weights = numpy.zeros(number_of_inputs + 1)

    def get_weights(self):
        return self.weights

    def set_threshold(self, threshold: int):
        self.threshold = threshold

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    # Prediction - predict the label base on the inputs
    def prediction(self, inputs):
        summation = self.weights[0] + inputs @ self.weights[1:]
        # activation
        if summation > 0:
            result = 1
        else:
            result = -1
        return result

    # Train - Improve the algorithm with new set of data, really poggers
    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.prediction(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


# Training Session
SimplePLAInstance = SimplePLA(2)
TrainingInputs = []
TrainingLabels = []

with open("train.txt") as TrainingDataFileStream:
    for line in TrainingDataFileStream:
        inputA, inputB, Label = line.split(',')
        inputs_TEMP = numpy.array([int(inputA), int(inputB)])
        label_TEMP = int(Label)
        # plotting, red O as 1, red X as -1
        if label_TEMP == 1:
            plt.plot([int(inputA)], [int(inputB)], 'ro')
        else:
            plt.plot([int(inputA)], [int(inputB)], 'rx')
        TrainingInputs.append(inputs_TEMP)
        TrainingLabels.append(label_TEMP)

SimplePLAInstance.train(TrainingInputs, TrainingLabels)

# Validating Session
with open("test.txt") as TestingDataFileStream:
    for line in TestingDataFileStream:
        inputA, inputB = line.split(',')
        inputs_TEMP = [int(inputA), int(inputB)]
        label_TEMP = SimplePLAInstance.prediction(inputs_TEMP)
        if label_TEMP == 1:
            plt.plot([int(inputA)], [int(inputB)], 'bo')
        else:
            plt.plot([int(inputA)], [int(inputB)], 'bx')

final_weight = SimplePLAInstance.get_weights()
print("The weights is : ")
print(final_weight)


# plotting the decision line
x_lin = numpy.linspace(-20, 20, 100)
plt.plot(x_lin, x_lin * -final_weight[1]/final_weight[2] + -final_weight[0]/final_weight[2])
plt.show()
