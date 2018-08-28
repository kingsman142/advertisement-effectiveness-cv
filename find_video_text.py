import cv2 as cv
import os
import glob
import json
from google.cloud import vision
from google.cloud.vision import types
client = vision.ImageAnnotatorClient()

OCR_DATA = {}

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

def extract_video_id(filename, batch_num):
    return filename.split("./batch" + batch_num)[1].split(".3gp")[0][1:]

def find_text(ids, base_dir):
    curr_id_num = 0
    for id in ids:
        i = 0
        video_text_data = []
        while True:
            curr_frame_filename = "./" + base_dir + "/" + id + "-" + str(i) + ".jpg"
            if not os.path.exists(curr_frame_filename):
                break
            with open(curr_frame_filename, "rb") as curr_frame:
                image_data = curr_frame.read()
                image = types.Image(content=image_data)
                response = client.text_detection(image=image)
                texts = response.text_annotations

                for text in texts:
                    text_description = text.description
                    x = text.bounding_poly.vertices[0].x
                    y = text.bounding_poly.vertices[0].y
                    width = text.bounding_poly.vertices[2].x - x
                    height = text.bounding_poly.vertices[2].y - y
                    bounding_box = [x, y, width, height]
                    text_data = [text_description, bounding_box]
                    video_text_data.append(text_data)
            i += 45
        OCR_DATA[id] = video_text_data
        if curr_id_num % 100 == 0:
            print(curr_id_num)
        curr_id_num += 1

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

find_text(batch1_ids, "batch1-frames")
print("Finished batch 1")
find_text(batch2_ids, "batch2-frames")
print("Finished batch 2")

with open("video_ocr_data.json", "w+") as ocr_file:
    ocr_file.write(json.dumps(OCR_DATA))
