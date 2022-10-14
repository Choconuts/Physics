import os
import cv2
import json

IMAGE_DIR = r"E:\BaiduNetdiskDownload\colmap_result"

with open(IMAGE_DIR + "\\transforms_train.json") as fp:
    meta = json.load(fp)


def select_image(frames):
    out = []
    for f in meta["frames"]:
        idx = int(f["file_path"][-8:-4])
        if idx % 14 == 3 or idx % 14 == 12:
            out.append(f)
    print(len(out))
    return out


for f in select_image(meta["frames"]):
    file = os.path.join(IMAGE_DIR, f["file_path"])
    img = cv2.imread(file)
    save = os.path.join("trains", f["file_path"])
    cv2.imwrite(save, img)


if __name__ == '__main__':
    pass
