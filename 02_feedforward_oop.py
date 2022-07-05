import numpy as np

def sigma(z):
    z /= 255
    return 1 / (1 + np.exp(-z))

with open('data_toy_problem/data_dark_bright_test_4000.csv') as f:
    data_list = f.readlines()

class Network:
    def __init__(self, num_of_input, num_of_hidden, num_of_output):
        self.input = num_of_input
        self.hidden = num_of_hidden
        self.output = num_of_output
        self.wa = np.random.rand(self.hidden, self.input) -0.5
        self.wb = np.random.rand(self.output, self.hidden) -0.5

    def feedforward(self, x):
        hidden = sigma(np.dot(self.wa, x))
        output = sigma(np.dot(self.wb, hidden))
        return (output)

    def test(self, data_list):
        count = 0 
        for line in data_list:
            x = line.split(",")
            target = int(x[0])

            for i in range(len(x)):
                x[i] = int(x[i])
            x = x[1:]
            output = self.feedforward(x)
            if output[0] > output[1] and target == 0 or  output[0] < output[1] and target == 1 :
                count += 1
        percent = 100 / len(data_list)*count
        return percent 


work = Network(4,3,2)

print(work.test(data_list))