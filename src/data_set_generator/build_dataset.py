import csv
import pandas as pd
import numpy as np



def get_lines(data_path, name):
    path = "{}/{}.csv".format(data_path, name)
    f = open(path)
    reader = csv.reader(f)
    header = next(reader)[1:] #skip the line with the header
    return (header, [line[1:] for line in reader])




def main():

    data_path = "../clean_features"
    split_ratio = {
        "train" : 0.6,
        "dev" : 0.2,
        "test" : 0.2
    }

    desired_data = [
        "bach",
        "beethoven",
        "brahms",
        "chopin",
        "debussy",
        "handel",
        "haydn",
        "liszt",
        "mozart",
        "vivaldi",
        "mendelssohn",
        "schubert",
        "telemann",
        "hummel",
        "dvorak",
    ]

    #just to extract the header, assume that the header is the same in all the files
    header,_  = get_lines(data_path, desired_data[0])
    header.append("y")
    all_data = [row + [y] for y, composer in enumerate(desired_data) for row in get_lines(data_path, composer)[1]]
    data_frame = pd.DataFrame(data = all_data, columns=header)
    

    #shuffle the data
    np.random.seed(41)
    data_frame = data_frame.reindex(np.random.permutation(data_frame.index))


    s = 0
    num_rows = data_frame.shape[0]
    for key in split_ratio:
        percent = split_ratio[key]
        i1 = round(s * num_rows)
        i2 = round((s + percent) * num_rows)
        sub_frame = data_frame.iloc[i1:i2,:]
        print("Saving {}, size = {}".format(key, i2 - i1))
        sub_frame.to_csv("{}.csv".format(key), index=False)
        s+=percent
    
if __name__ == "__main__":
    main()
