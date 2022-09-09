

tag = 'val'
image_dir = r'G:\WorkSpace\Houdini-Projects\vellumfluidsexamples_houdini_19\VellumFluidsExamples' + f'\\{tag}'

out_dir = 'data'


import os
import json
import shutil


for file in os.listdir(image_dir):
    ckpt = file + '.mantra_checkpoint'
    ckpt = os.path.join(image_dir, ckpt)
    if os.path.exists(ckpt):
        os.remove(ckpt)
        os.remove(os.path.join(image_dir, file))

with open(f'transforms_{tag}.json', 'r') as fp:
    meta = json.load(fp)


out_meta = {
    'camera_angle_x': meta['camera_angle_x'],
    'frames': []
}


for i in range(100):
    file = f'r_{i}.png'
    frame = meta['frames'][i]
    path = frame['file_path']
    print(file, path)

    save_path = os.path.join(out_dir, path + file[-4:])

    shutil.copy(os.path.join(image_dir, file), save_path)
    out_meta['frames'].append(frame)


with open(os.path.join(out_dir, f'transforms_{tag}.json'), 'w') as fp:
    json.dump(out_meta, fp, indent=4)


if __name__ == '__main__':
    pass
