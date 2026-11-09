import numpy as np
import pandas as pd
import sklearn.model_selection as skms
import random as rand

def dataset_setup(csv_path):   
    raw_df = pd.read_csv(csv_path)

    #Randomly select the 5 classes to keep
    rng = np.random.default_rng(39403947)
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


    

def main():
    x, y = dataset_setup("covtype-10.csv")
    
    #train and test data split
    x_train, x_test, y_train, y_test = skms.train_test_split(x, y, test_size=0.2)
    


if __name__ == "__main__":
    main()

