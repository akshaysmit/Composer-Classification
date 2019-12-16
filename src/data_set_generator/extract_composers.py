import csv
import json
import os
import shutil
import argparse


DEFAULT_INDEX = "../../ClassicalArchives-MIDI-Collection/"
DEFAULT_ARG = "./extraction_arguments.json"
DEFAULT_OUT = "./out"

def get_args(file_path):
    f = open(file_path)
    return json.load(f)
    

def get_lines():
    path = "./cleaned_catalog.csv"
    f = open(path)
    reader = csv.reader(f)
    next(reader) #skip the line with the header
    return [line for line in reader]

def extract_composors(lines, composors):
    result = {}
    for line in lines:
        composer = line[0].lower()
        if composer in composors:
            file_path = line[2]
            if "http" in file_path: #some rows reference to a url
                continue
            if composer in result:
                result[composer].append(file_path)
            else:
                result[composer] = [file_path]
    return result


def path_name(output_path, composer):
    return "{}/{}".format(output_path, composer)

def create_new_directories(output_path, composors):
    if os.path.isdir(output_path):
        try:
            shutil.rmtree(output_path)
        except OSError as e:
            print("Error when trying to remove directory %s", e)

    try:
        os.mkdir(output_path)
        for composer in composors:
            try:
                os.mkdir(path_name(output_path, composer))
            except OSError:
                print("Error when creating sub directory")
    except OSError:
        print("Error when creating directory")
    
        

def copy_files(files_by_composors, output_path, base_data_path):
    copied = {key : 0 for key in files_by_composors.keys()}
    for composer in files_by_composors.keys():
        for file in files_by_composors[composer]:
            from_file = "{}/{}".format(base_data_path, file)
            to_file = "{}/".format(path_name(output_path, composer))
            try:
                shutil.copy2(from_file,to_file)
                copied[composer]+=1
            except OSError:
                pass
    return copied


def extract_files(index_base = DEFAULT_INDEX,
                arg_path = DEFAULT_ARG,
                output_path = DEFAULT_OUT,
                verbuse = True):
    args = get_args(arg_path)
    lines = get_lines()
    files_by_composors = extract_composors(lines, args["composer_names"])
    create_new_directories(output_path, files_by_composors.keys())
    res = copy_files(files_by_composors, output_path, index_base)
    if verbuse:
        for key in files_by_composors.keys():
            print("{} songs by {} copied".format(res[key], key))
        print("Total number of songs = {}".format(sum(res.values())))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX)
    parser.add_argument("--arg_path", type=str, default=DEFAULT_ARG)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()
    extract_files(args.index_path, args.arg_path, args.out)
    
