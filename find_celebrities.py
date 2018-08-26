import cv2 as cv
import os
import glob
import json

NUM_CELEBS_DATA = {}

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

def extract_video_id(filename, batch_num):
    return filename.split("./batch" + batch_num)[1].split(".3gp")[0][1:]

def find_num_celebrities(ids, base_dir):
    for id in ids:
        i = 0
        num_celebrities = 0
        while True:
            curr_frame_filename = "./" + base_dir + "/" + id + "-" + str(i) + ".jpg"
            if not os.exists(curr_frame_filename):
                break
            with open(curr_frame_filename, "rb") as curr_frame:
                image_data = curr_frame.read()
                # TODO: Somehow connect this to AWS Rekognition for Celebrity Detection.  We don't have an AWS account currently
                num_celebrities += -1 # TODO: Edit this
            i += 1
        NUM_CELEBS_DATA[id] = num_celebrities

effective_data_keys = effective_data.keys()

batch1_filenames = glob.glob("./batch1/*.3gp")
batch1_filenames = [filename for filename in batch1_filenames if extract_video_id(filename, "1") in effective_data_keys]
batch1_ids = [extract_video_id(filename, "1") for filename in batch1_filenames]

batch2_filenames = glob.glob("./batch2/*.3gp")
batch2_filenames = [filename for filename in batch2_filenames if extract_video_id(filename, "2") in effective_data_keys]
batch2_ids = [extract_video_id(filename, "2") for filename in batch2_filenames]
batch2_ids = [id for id in batch2_ids if id not in batch1_ids]

print("Batch 1 size: %d" % len(batch1_ids))
print("Batch 2 size: %d" % len(batch2_ids))

find_num_celebrities(batch1_ids, "batch1-frames")
print("Finished batch 1")
find_num_celebrities(batch2_ids, "batch2-frames")
print("Finished batch 2")

with open("video_num_celebs.json", "w+") as num_celebs_file:
    num_celebs_file.write(json.dumps(NUM_CELEBS_DATA))
