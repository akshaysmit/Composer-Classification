import argparse
import pandas as pd

DEFAULT_INDEX = "../../ClassicalArchives-MIDI-Collection/CSV-Index/classical_archives_index.csv"

replace_pairs = [[' fuge ', ' fugue '], [' fuga ', ' fugue '] ,[' fantasie ', ' fantasy '] , \
                 [' fantasia ', ' fantasy '], [' et ', ' and '],  [' ', '']]

def process_index(path):
    df = pd.read_csv(path)
    df['title'] = df['title'].str.lower()

    for i in range(len(replace_pairs)):
        df['title'] = df['title'].str.replace(replace_pairs[i][0], replace_pairs[i][1])
    
    print('Original number of rows is: ', df.shape[0])
    df.drop_duplicates(subset='title', inplace=True)
    print('New number of rows is: ', df.shape[0])
    df.to_csv('cleaned_catalog.csv', index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX)
    args = parser.parse_args()
    process_index(args.index_path)
