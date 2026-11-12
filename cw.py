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

    cov_type_df = cov_type_df.reset_index(drop=True) #fixes indices

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

    #Creates hidden layer between inputs and outputs
    #Each subsequent call adds another hidden layer in between the previously created hidden layer and output layer
    def add_hidden_layer(self, n_nodes: int):
        
        #Creates input biases and weights if no other hidden layers exist yet
        if not self.hl_nodes:
            input_biases = np.zeros(n_nodes)
            input_weights = rng.standard_normal((self.n_inputs, n_nodes)) * 0.1

            #This in only done for the first layer, 
            #every layer after the first uses the previous layer's outputs as it's inputs
            self.biases.append(input_biases)
            self.weights.append(input_weights)
        else:
            #If a hidden layer already exists...
            #Overwrite the previous hidden layer's output weights (this layer's input weights) and output biases (this layer's input biases)
            self.biases[-1] = np.zeros(n_nodes)
            self.weights[-1] = rng.standard_normal((self.hl_nodes[-1], n_nodes)) * 0.1

        #Creates output weights and biases (may be overwritten if another layer is added)
        output_biases = np.zeros(self.n_outputs)
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

    def prediction():
        pass



def main():
    x, y = dataset_setup("covtype-10.csv")
    
    #train and test data split
    x_train, x_test, y_train, y_test = skms.train_test_split(x, y, test_size=0.2)

    test_mlp = MLP(3, 1)
    print("Just inputs and outputs")
    test_mlp.print_mlp()
    print("Now w/ hidden layer")
    test_mlp.add_hidden_layer(2)
    test_mlp.print_mlp()
    print("Adding another hidden layer")
    test_mlp.add_hidden_layer(2)
    test_mlp.print_mlp()


if __name__ == "__main__":
    main()

