import json
from autocorrect import spell

with open("./annotations_videos/video/cleaned_result/video_Effective_clean.json", "r") as EFFECTIVE_DATA_FILE:
    effective_data = json.loads(EFFECTIVE_DATA_FILE.read())

with open("./video_ocr_data.json", "r") as OCR_FILE:
    ocr_data = json.loads(OCR_FILE.read())

OCR_DATA_CLEAN = {}

i = 0
for video_id, video_text_data in ocr_data.items():
    words = set()
    for text_info in video_text_data:
        clean_text = text_info[0].replace("\n", " ").lower().encode('ascii', 'ignore').decode('utf-8')
        clean_text = ''.join(ch for ch in clean_text if ch.isalnum() or ch in [" ", "-", "'"])
        new_words = clean_text.split()
        for word in new_words:
            word = spell(word)
            words.add(word)
    words = list(words)
    OCR_DATA_CLEAN[video_id] = words
    i += 1
    print(i)

with open("video_ocr_data_clean_words.json", "w+") as ocr_file:
    ocr_file.write(json.dumps(OCR_DATA_CLEAN))
