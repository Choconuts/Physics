
import torch
import json

start = 75
number = 25
tag = 'val'

with open('template.py', 'r') as fp:
    lines = fp.readlines()

with open(f'transforms_{tag}.json', 'r') as fp:
    obj = json.load(fp)

mat_strs = []
for frame in obj['frames']:
    mat = frame['transform_matrix']
    mat = torch.tensor(mat)
    mat = mat.t()
    mat_str = mat.__repr__()
    mat_str = mat_str.replace('tensor(', '').replace(')', '').replace(' ', '').replace(',', ', ')
    mat_strs.append(mat_str)

mat_strs = mat_strs[start: start + number]

for i, line in enumerate(lines):
    if '$placeholder$' in line:
        lines[i] = ',\n \n'.join(mat_strs)
    if '$skip$' in line:
        lines[i] = line.replace('$skip$', f'{start}')
    if '$tag$' in line:
        lines[i] = line.replace('$tag$', f'{tag}')

with open('script.py', 'w') as fp:
    fp.writelines(lines)


if __name__ == '__main__':
    pass
