import cv2 as cv
import os
import glob
import json

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

def extract_video_id(filename, batch_num):
    return filename.split("./batch" + batch_num)[1].split(".3gp")[0][1:]

def extract_video_frames(video_id, output_folder, batch_num):
    filename = "./batch" + batch_num + "/" + video_id + ".3gp"
    capture = cv.VideoCapture(filename)
    video_len = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: %d" % video_len)
    i = 0
    step_size = 5
    while capture.isOpened():
        if i % step_size == 0:
            ret, frame = capture.read()
            new_filename = ".\\" + output_folder + "\\" + video_id + "-" + str(int(i/step_size)) + ".jpg"
            cv.imwrite(new_filename, frame)
        i += 1
        if i >= video_len:
            capture.release()

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

count = 1
for video_id in batch2_ids:
    extract_video_frames(video_id, "batch2-frames", "2")
    print("Done with video %d" % count)
    count += 1
