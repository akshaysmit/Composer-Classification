import unittest
import numpy as np
import pandas as pd
import json

from pprint import pprint

from sklearn.preprocessing import StandardScaler


def get_indices(frame, col, seq):
    if len(seq) > 1:
        return np.concatenate(np.array([np.where(frame[col] == i)[0] for i in seq]))
    else:
        return np.where(frame[col] == seq[0])[0]


def remove_labels(data, label_col, label_values):
    indices = get_indices(data, label_col, label_values)
    return data.drop(indices)

def remove_nan(data):
        return data[~np.isnan(data).any(axis=1)]


def group_data(groups, data, col):
    new_frames = []
    for i, group in enumerate(groups):
        indices = get_indices(data, col, group)
        data_i = data.iloc[indices,:].copy()
        data_i.loc[:,col] = i
        new_frames.append(data_i)
    result = pd.concat(new_frames, axis = 0)
    result = result.reindex(np.random.permutation(result.index))
    return result

def get_y_labels():
    extraction_arg_path = "../data_cleaning/extraction_arguments.json"
    f = open(extraction_arg_path)
    return json.load(f)["composer_names"]

def get_features(name):
    path = "./{}.txt".format(name)
    f = open(path)
    result = [line.strip('\n') for line in f.readlines() if line[0] != "#"]
    return result
        

def read_and_clean(standardize=True, num_labels=15, same_size=False, group_composers = False, selected_features = None):
    base_path = "../data_set_generator/"
    drop_cols = ["y", "Note_Density_per_Quarter_Note_Variability"]
    train_data, dev_data, test_data = [pd.read_csv(f) for f in ["{}{}.csv".format(
        base_path, name) for name in ["train", "dev", "test"]]]

    value_counts = train_data['y'].value_counts()

    num_different_y = value_counts.shape[0]
    new_mapping = None  # Maps old labels to new labels
    if num_labels < num_different_y:

        label_rem = train_data["y"].value_counts().index[num_labels:]
        label_sel = train_data["y"].value_counts().index[:num_labels]

        # Remove all rows with different label than those in label_sel
        train_data = remove_labels(train_data, "y", label_rem)
        dev_data = remove_labels(dev_data, "y", label_rem)

        new_mapping = {val: i for i, val in enumerate(label_sel)}

        # Apply new mapping
        map_fn = lambda x: new_mapping[x]
        train_data['y'] = train_data['y'].apply(map_fn)
        dev_data['y'] = dev_data['y'].apply(map_fn)

    if selected_features is not None:
        if selected_features not in ['melodic', 'vertical', 'rhythm', 'texture']:
            print("ERROR: selected_features must be melodic, vertical, rhythm or texture")
            quit()
        features = get_features(selected_features) + ["y"]
        train_data = train_data[features]
        dev_data = dev_data[features]
        drop_cols = ["y"]    

        
    if group_composers:
        ERAS = [["bach", "handel", "vivaldi","telemann"], ["haydn","mozart"], ["brahms","chopin", "debussy","liszt","mendelssohn", "dvorak"]]
        y_labels = {name : i for i,name in enumerate(get_y_labels())}
        groups = []
        for era in ERAS:
            groups.append([y_labels[name] for name in era])

        train_data = group_data(groups, train_data, "y")
        dev_data = group_data(groups, dev_data, "y")

    
    if same_size:
        value_counts = train_data['y'].value_counts()
        largest_number = np.max(value_counts.iloc[:])
        for i in range(num_labels):
            indices = np.where(train_data['y'] == i)[0]

            diff = largest_number - value_counts.loc[i]
            if diff > 0:
                sampled_indices = np.random.choice(indices, diff)
                rows = train_data.iloc[sampled_indices, :]
                train_data = train_data.append(rows)

        np.random.shuffle(train_data.values)

    train_y, dev_y = [data['y'] for data in [train_data, dev_data]]
    train_X, dev_X = [data.drop(drop_cols, axis=1)
                      for data in [train_data, dev_data]]
 
    # flatten y as numpy array
    train_y = np.ndarray.flatten(train_y.to_numpy())
    dev_y = np.ndarray.flatten(dev_y.to_numpy())


    headers = train_X.columns

    
    #Remove NAN
    train_X = train_X.to_numpy()
    train_X = train_X.astype(np.float64)
    train_y = train_y[~np.isnan(train_X).any(axis=1)]
    train_X = train_X[~np.isnan(train_X).any(axis=1)]
    dev_X = dev_X.to_numpy()
    dev_X = dev_X.astype(np.float64)
    dev_y = dev_y[~np.isnan(dev_X).any(axis=1)]
    dev_X = dev_X[~np.isnan(dev_X).any(axis=1)]


    if standardize:
        train_X, dev_X = [StandardScaler().fit_transform(data)
                          for data in [train_X, dev_X]]

    return (train_X, train_y, dev_X, dev_y, headers)


class LoadDataTest(unittest.TestCase):

    def setUp(self):
        self.test_frame = pd.DataFrame([[1, 2, 0],
                                        [2, 4, 1],
                                        [3, 6, 1],
                                        [4, 8, 0],
                                        [5, 10, 2]],
                                         columns=["x1", "x2", "y"])

    def test_get_indices(self):
        input_output = (([0], [0, 3]),
                        ([0, 1], [0, 1, 2, 3]),
                        ([0, 2], [0, 3, 4]))
        for in_val, output in input_output:
            result = np.sort(get_indices(self.test_frame, "y", in_val))
            self.assertTrue(np.array_equal(result, output))

    def test_remove_labels(self):
        input_output = (([0], [[2, 4, 1], [3, 6, 1], [5,10,2]]),
                        ([2], [[1, 2, 0], [2,4,1], [3,6,1], [4,8,0]]),
                        ([1,2], [[1,2,0], [4,8,0]]))

        for in_val, output in input_output:
            result = remove_labels(self.test_frame, "y", in_val)
            self.assertTrue(np.array_equal(result.to_numpy(), np.array(output)))

    def test_remove_nan(self):
        arr = np.array([[np.nan, 0], [0,1], [0, np.nan], [1,0]])
        result = remove_nan(arr)
        self.assertTrue(np.array_equal(result, np.array([[0,1], [1,0]])))


if __name__ == "__main__":

    #unittest.main()
    train_X, train_y, dev_X, dev_y, headers= read_and_clean(num_labels=15, same_size=False, group_composers=True, use_selected_features=True)
    #print(train_y, dev_y)
    print(train_X.shape, dev_X.shape)

