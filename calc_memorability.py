import json
import glob
import os
from scipy import misc
from skimage import io, transform
import numpy as np
import caffe

def extract_video_id(filename, batch_num):
    return filename.split("./batch" + batch_num)[1].split(".3gp")[0][1:]

def calc_video_avg_memorability(video_id, batch_num, net):
    j = 0
    avg_memorability = 0
    base_filename = "./batch" + batch_num + "-frames/" + video_id + "-"
    while os.path.exists(base_filename + str(j) + ".jpg"):
        if j % 10 == 0:
            '''img = io.imread(base_filename + str(j) + ".jpg", as_grey = True)
            img = transform.resize(img, (227, 227))
            net.blobs['data'].data[...] = img
            net.forward()
            memorability = net.blobs['fc8-euclidean'].data[0][0]
            print memorability
            avg_memorability += memorability'''
        j += 1
    return (avg_memorability / j)

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

mems = {}
print "HERE"

memnet = caffe.Net('./memnet/deploy.prototxt', 1, weights='./memnet/memnet.caffemodel')

i = 0
for video_id in effective_data.keys():
    if video_id in batch1_ids:
        avg_memorability = calc_video_avg_intensity(video_id, "1", memnet)
        mems[video_id] = avg_memorability
    elif video_id in batch2_ids:
        avg_memorability = calc_video_avg_intensity(video_id, "2", memnet)
        mems[video_id] = avg_memorability
    #print("%s: %.3f" % (video_id, average_intensities[video_id]))
    i += 1
    if i % 100 == 0:
        print i

with open("video_average_memorabilities.json", "w+") as avg_mem_file:
    avg_mem_file.write(json.dumps(mems))
