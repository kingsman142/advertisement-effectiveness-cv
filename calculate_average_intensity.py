import json
import glob
import os
from scipy import misc
from skimage import io
import numpy as np

def extract_video_id(filename, batch_num):
    return filename.split("./batch" + batch_num)[1].split(".3gp")[0][1:]

def calc_video_avg_intensity(video_id, batch_num):
    j = 0
    avg_intensity = 0
    base_filename = "./batch" + batch_num + "-frames/" + video_id + "-"
    while os.path.exists(base_filename + str(j) + ".jpg"):
        #print("dir: " + "./batch1-frames/" + video_id + "-" + str(j) + ".jpg")
        #img = misc.imread("./batch1-frames/" + video_id + str(j) + ".jpg", 'L')
        #avg_intensity += calc_avg_intensity(img)
        img = io.imread(base_filename + str(j) + ".jpg", as_grey = True)
        avg_intensity += np.mean(img)
        j += 1
    return (avg_intensity / j)

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

effective_data_keys = effective_data.keys()

batch1_filenames = glob.glob("./batch1/*.3gp")
batch1_filenames = [filename for filename in batch1_filenames if extract_video_id(filename, "1") in effective_data_keys]
batch1_ids = [extract_video_id(filename, "1") for filename in batch1_filenames]

batch2_filenames = glob.glob("./batch2/*.3gp")
batch2_filenames = [filename for filename in batch2_filenames if extract_video_id(filename, "2") in effective_data_keys]
batch2_ids = [extract_video_id(filename, "2") for filename in batch2_filenames]
batch2_ids = [id for id in batch2_ids if id not in batch1_ids]

average_intensities = {}

i = 0
for video_id in effective_data.keys():
    if video_id in batch1_ids:
        avg_intensity = calc_video_avg_intensity(video_id, "1")
        average_intensities[video_id] = (avg_intensity / j)
    elif video_id in batch2_ids:
        avg_intensity = calc_video_avg_intensity(video_id, "2")
        average_intensities[video_id] = (avg_intensity / j)
    #print("%s: %.3f" % (video_id, average_intensities[video_id]))
    i += 1
    if i % 100 == 0:
        print(i)

with open("video_average_intensities.json", "w+") as avg_intensity_file:
    avg_intensity_file.write(json.dumps(average_intensities))
