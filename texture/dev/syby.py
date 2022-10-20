import cv2
import numpy as np


data_path = r"C:\Users\MSI\Desktop\Tmp\out_texture.txt"
image_path = r"C:\Users\MSI\Desktop\Tmp\out_texture.png"
with open(data_path, "rb") as fp:
    data = fp.read()
    arr = np.frombuffer(data, np.float32, 1280 * 720 * 4)
    print(arr)
    cv2.imwrite(image_path, arr.reshape([720, 1280, 4]) * 255)


if __name__ == '__main__':
    pass
