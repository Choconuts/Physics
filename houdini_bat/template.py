import hou
import os

transforms = [
$placeholder$
]

cam = hou.node('obj/cam1')
out = hou.node('out/mantra_ipr')

for i, trans in enumerate(transforms):
    idx = i + $skip$
    if trans is None:
        continue
    filename = f'$HIP/$tag$/r_{idx}.png'
    mat = hou.Matrix4(trans)
    cam.setWorldTransform(mat)
    hou.node('out/mantra_ipr')
    out.render(output_file=filename)


