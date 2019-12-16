import mido
import csv
import numpy as np
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random


OUT_BASE = "train_images"
DEFAULT_INDEX = "../../ClassicalArchives-MIDI-Collection/"


def augment_data_random(data,  k):
    new_data = [data]

    for i in range(k):
        new = np.zeros_like(data)
        dx = np.random.randint(-150, 150)
        dy = np.random.randint(-15, 15)
        
        new = np.roll(data, dy, axis=0)
        if dy >= 0:
            new[0:dy,:] = 0
        else:
            new[dy:,:] = 0

        new = np.roll(new, dx, axis=1)
        if dx >= 0:
            new[:,0:dx] = 0
        else:
            new[:,dx:] = 0
        

        new_data.append(new)
    return new_data  

def augment_data(data):
    pitch_up = np.zeros_like(data)
    pitch_down = np.zeros_like(data)

    pitch_down_half = np.zeros_like(data)
    pitch_up_half = np.zeros_like(data)

    shift_right = np.zeros_like(data)
    shift_left = np.zeros_like(data)

    shift_right_2 = np.zeros_like(data)
    shift_left_2 = np.zeros_like(data)

    pitch_up[12:, :] = data[:-12, :]
    pitch_down[:-12, :] = data[12:, :]

    pitch_up_half[6:, :] = data[:-6, :]
    pitch_down_half[:-6, :] = data[6:, :]

    shift_right[:, 50:] = data[:, :-50]
    shift_left[:, :-50] = data[:, 50:]

    shift_right_2[:, 100:] = data[:, :-100]
    shift_left_2[:, :-100] = data[:, 100:]

    return [data, pitch_down, pitch_up, shift_right, shift_left, pitch_down_half, pitch_up_half, shift_left_2, shift_right_2]


MIN_note = 128
MAX_note = 0
dist = np.zeros(128)


def extract_song(mid):
    global MIN_note
    global MAX_note
    FS = 10
    time = 155
    out_sample = np.zeros([128, FS * time])
    "print comes here"
    merged = mido.merge_tracks(mid.tracks)

    tempo = 500000
    current_time = 0
    current_sample = 0
    current_intensity = np.zeros(128)
    for msg in merged:
        if msg.time > 0:
            delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        else:
            delta = 0
        if hasattr(msg, "note"):
            MIN_note = min(msg.note, MIN_note)
            MAX_note = max(msg.note, MAX_note)
            dist[msg.note] += 1
            if msg.type == "note_on":
                current_intensity[msg.note] = msg.velocity
            else:
                current_intensity[msg.note] = 0
        last_time = current_time
        current_time += delta

        if current_time > time:
            break

        new_sample = np.floor(current_time * FS).astype(int)

        if new_sample > current_sample:
            new_sample = np.floor(current_time * FS).astype(int)
            block = np.tile(current_intensity.reshape(
                128, 1), new_sample - current_sample)
            out_sample[:, current_sample:new_sample] = block
            current_sample = new_sample

        if hasattr(msg, "tempo"):
            tempo = msg.tempo

    return out_sample


def main():

    split_ratio = [0.6, 0.8, 1]

    f = open("./extraction_arguments.json")
    names = json.load(f)["composer_names"]

    y_map = {name.upper(): i for i, name in enumerate(names)}

    f = open("cleaned_catalog.csv")
    reader = csv.reader(f)
    next(reader)
    lines = [l for l in reader if l[0] in y_map]
    random.shuffle(lines)

    name_file = open("train_data.csv", "w")
    writer = csv.writer(name_file)
    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)
   
    for i, line in enumerate(lines):
        print("Reading {}".format(i))
        try:
            mid = mido.MidiFile("{}/{}".format(DEFAULT_INDEX, line[2]))
            data = extract_song(mid)
            which_set = 0
            if i > len(lines) * split_ratio[0]:
                which_set = 1 if i < len(lines) * split_ratio[1] else 2
            name = "song_{}".format(i)
            plt.gray()
            plt.imsave("{}/{}.png".format(OUT_BASE, name),
                       data, vmin=0, vmax=128)
            writer.writerow([y_map[line[0]], name, which_set])
        except:
            print("err")

    print(MIN_note, MAX_note)
    x = np.linspace(0,128,128)
    plt.plot(x, dist)
    plt.show()
    print(dist)


if __name__ == "__main__":
    main()


            #if i < len(lines) * split_ratio[0]:
            #    for j, out_data in enumerate(augment_data_random(data, K)):
            #        name = "song_{}".format(i*K + j)
            #        which_set = 0  # to indicate train set
            #        plt.gray()
            #        plt.imsave("{}/{}.png".format(OUT_BASE, name),
            #                   out_data, vmin=0, vmax=128)
            #        writer.writerow([y_map[line[0]], name, which_set])