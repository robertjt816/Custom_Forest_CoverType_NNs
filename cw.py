import numpy as np
import pandas as pd
import sklearn.model_selection as skms

rng = np.random.default_rng(39403947)

def dataset_setup(csv_path):   
    raw_df = pd.read_csv(csv_path)

    #Randomly select the 5 classes to keep
    
    chosen_classes = rng.choice(7, size=5, replace=False)
    chosen_classes = chosen_classes + 1
    raw_df = raw_df[raw_df["Cover_Type"].isin(chosen_classes)]

    #Create new df only containing 2500 samples of the each of the chosen_classes
    cov_type_df = pd.DataFrame()
    for i in chosen_classes:
        same_class_df = raw_df[raw_df["Cover_Type"] == i]
        rng_sample = same_class_df.sample(n=2500, random_state=rng)
        cov_type_df = pd.concat([cov_type_df, rng_sample])

    #Splits dfs into inputs (x) and ground truths (y)
    x = cov_type_df.drop(columns=["Cover_Type"])
    y = cov_type_df["Cover_Type"]

    return x, y



class MLP:
    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hl_nodes = [] #Contains a list of the number of nodes in each hidden layer
        self.weights = [] 
        self.biases = []
        self.weighted_sums = [] #populated in prediction func, for use in backprop
        self.activations = [] #populated in prediction func, for use in backprop

    #Creates hidden layer between inputs and outputs
    #Each subsequent call adds another hidden layer in between the previously created hidden layer and output layer
    def add_hidden_layer(self, n_nodes: int):
        
        #Creates input biases and weights if no other hidden layers exist yet
        if not self.hl_nodes:
            input_biases = np.zeros((1, n_nodes))
            input_weights = rng.standard_normal((self.n_inputs, n_nodes)) * 0.1

            #This in only done for the first layer, 
            #every layer after the first uses the previous layer's outputs as it's inputs
            self.biases.append(input_biases)
            self.weights.append(input_weights)
        else:
            #If a hidden layer already exists...
            #Overwrite the previous hidden layer's output weights (this layer's input weights) and output biases (this layer's input biases)
            self.biases[-1] = np.zeros((1, n_nodes))
            self.weights[-1] = rng.standard_normal((self.hl_nodes[-1], n_nodes)) * 0.1

        #Creates output weights and biases (may be overwritten if another layer is added)
        output_biases = np.zeros((1, self.n_outputs))
        output_weights = rng.standard_normal((n_nodes, self.n_outputs)) * 0.1

        #Appends new nodes, biases, and weights to the MLP object's lists
        self.hl_nodes.append(n_nodes)
        self.biases.append(output_biases)
        self.weights.append(output_weights)

    def print_mlp(self):
        print(f"Number of inputs: {self.n_inputs}")
        print(f"hl nodes list: {self.hl_nodes}")
        print(f"Biases list: {self.biases}")
        print(f"Weights list: {self.weights}")
        print(f"Number of outputs: {self.n_outputs}")

    def prediction(self, x, activation_funcs):
        for weights, biases, activation in zip(self.weights, self.biases, activation_funcs):
            self.weighted_sums.append(np.dot(x, weights)+biases) #Need each layer's weighted sum for backprop
            
            x = activation(np.dot(x, weights) + biases)

            self.activations.append(x) #Need each layer's activation func output for backprop

        return x, np.argmax(x, axis=1) #Outputs list of all 5 classes final activation output and the list of the indices of the highest value outputs (the class prediction)

    #x format:
    # [[1. 0. 0. 0. 0.]
    # [0. 1. 0. 0. 0.]
    # [0. 0. 1. 0. 0.]
    # [0. 0. 0. 1. 0.]
    # [0. 0. 0. 0. 1.]]
    #np.argmax(x, axis=1) format:
    # [4 4 4 0 4]
    

    def baseline_backpropogate(self, input, prediction, ground_truth, n_samples, learning_rate): #Backpropagation func for set single hidden layer
        #2nd layer of dvs
        dl_dws2 = (prediction - ground_truth) * MLP.sigmoid_dv(self.weighted_sums[1]) #The derivative of the loss with respect to the second weighted sum (the final ws)
        dl_dw2 = np.dot(self.activations[0].T,dl_dws2)/n_samples #...
        dl_db2 = np.sum(dl_dws2, axis=0, keepdims=True)/n_samples
        #1st layer of dvs
        dl_da1 = np.dot(dl_dws2,self.weights[1].T)
        dl_dws1 = dl_da1 * MLP.relu_dv(self.weighted_sums[0])
        dl_dw1 = np.dot(input.T,dl_dws1) / n_samples
        dl_db1 = np.sum(dl_dws1, axis=0, keepdims=True)/n_samples
        #Update weights
        self.biases[0] -= learning_rate * dl_db1
        self.weights[0] -= learning_rate * dl_dw1
        self.biases[1] -= learning_rate * dl_db2
        self.weights[1] -= learning_rate * dl_dw2

        self.weighted_sums.clear() #Clear weighted_sums list for next prediction cycle to populate it
        self.activations.clear() #Clear activations list for next prediction cycle to populate it

    @staticmethod
    def loss(ground_truth, prediction): #Using MSE
        return np.mean((ground_truth - prediction) ** 2) #Averages across each samples' row and then across all samples

    @staticmethod
    def accuracy(ground_truth, predicted_classes):
        correct_count = 0
        for gt_val, predicted_class in zip(ground_truth, predicted_classes):
            if gt_val == predicted_class:
                correct_count += 1
        return correct_count/len(ground_truth)

    
        
    #Activation funcs
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    @staticmethod
    def relu_dv(x):
        return np.where(x>0, 1, 0)
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_dv(x):
        s = MLP.sigmoid(x)
        return s * (1 - s)
    
#Size of input values caused predictions blow up in sigmoid
#Sample prediction before standardization: [1.         1.         0.99999995 1.         1.        ]
def standardization(input_data): #z-score normalization
    mean = np.mean(input_data, axis=0)
    std = np.std(input_data, axis=0)

    return (input_data - mean)/std

#Fitting 1d ground truth list to the 5 node output of the baseline MLP
# +Converts ground truth list to values 0-4 instead of random 5 from 1-7
def y_shaping(y, n_outputs):
    classes = np.unique(y) #Returns list of possible classes ([3, 4, 5, 6, 7] for this seed)
    adjusted_indices = np.searchsorted(classes, y) #Converts y's gt classes to 0-4 instead of random 5 from 1-7
    gt = np.zeros((len(y), n_outputs))
    row_indices = np.arange(len(y)) #Creates a list from 0 to y
    gt[row_indices, adjusted_indices] = 1
    return gt, adjusted_indices #First return val: formatted for MLP, Second return val: formatted for accuracy measure

def main():
    x, y = dataset_setup("covtype-10.csv")
    
    #train and test data split
    x_train, x_test, y_train, y_test = skms.train_test_split(x, y, test_size=0.2, shuffle=False)

    y_train, y_train_indices_updated = y_shaping(y_train, 5)
    y_test, y_test_indices_updated= y_shaping(y_test, 5)

    x_train = standardization(x_train)
    x_test = standardization(x_test)

    baseline_mlp = MLP(10, 5)
    baseline_mlp.add_hidden_layer(2)


    for i in range(5000):
        prediction, predicted_classes = baseline_mlp.prediction(x_train, [MLP.relu, MLP.sigmoid])
        if i % 100 == 0:
            print(f"loss: {MLP.loss(y_train, prediction)}")
            print(f"accuracy: {MLP.accuracy(y_train_indices_updated, predicted_classes)}")
        baseline_mlp.baseline_backpropogate(x_train, prediction, y_train, len(x_train), 0.1)
    
    


    



    #Note: Make sure to clear weighted_sums and activations lists from the MLP object after backprop finishes

if __name__ == "__main__":
    main()

