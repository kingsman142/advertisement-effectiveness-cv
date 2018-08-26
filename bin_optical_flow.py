import os
import glob
import json

optical_files = glob.glob("./Optical-Flow/*.txt")
print("Files: %d" % len(optical_files))

def calc_min_max_magnitudes():
    low = None
    high = None
    count = 0
    optical_bins = {}

    for file in optical_files:
        video_id = file.split("/")[2].split(".")[0]
        with open(file) as optical_data:
            for cnt, line in enumerate(optical_data):
                magnitude = float(line)
                if low is None or magnitude < low:
                    low = magnitude
                if high is None or magnitude > high:
                    high = magnitude
        count += 1
        if count % 100 == 0:
            print(count)
    print("Low: %.4f" % (low))
    print("High: %.4f" % (high))

def bin_magnitudes_per_video():
    count = 0
    bins_per_video = {}

    for file in optical_files:
        video_id = file.split("/")[2].split(".")[0]
        bins = [0] * 30
        with open(file) as optical_data:
            for cnt, line in enumerate(optical_data):
                magnitude = float(line)
                bin_num = int(magnitude / 766667) # 766667 is the size of each bin: minimum magnitude = 0, max = 23,000,010
                bins[bin_num] += 1
        bins = [float(x) / sum(bins) for x in bins]
        bins_per_video[video_id] = bins
        count += 1
        if count % 100 == 0:
            print(count)
    return bins_per_video

bins_per_video_data = bin_magnitudes_per_video()

with open("optical_flow_bins.json", "w+") as OF_data:
    print("Saving data to file")
    json.dump(bins_per_video_data, OF_data)
