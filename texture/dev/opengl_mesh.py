import numpy as np
import trimesh
from OpenGL.GL import *
import glfw
from visilab.rasterization.pipeline import MeshFilter, Buffer, Shader
from visilab.rasterization.camera import Camera
from visilab.rasterization.window import GLFWWindow
from visilab.rasterization.renderer import Renderer
from visilab.rasterization.frame import FrameBuffer


vs = """
     #version 330
     in vec3 position;
     in vec3 color;
     out vec3 newColor;
     void main()
     {
         gl_Position = vec4(position * 2 - 1, 1.0f);
         newColor = color;
     }
     """

fs = """
     #version 330
     in vec3 newColor;
     out vec4 outColor;
     void main()
     {
         outColor = vec4(newColor, 1.0f);
     }
     """

window = GLFWWindow(1024, 1024)
window.init()
glEnable(GL_DEPTH_TEST)
mesh = trimesh.load("lego_uv3.obj")
uvw = np.concatenate([mesh.visual.uv, np.zeros_like(mesh.visual.uv[..., :1])], -1)
pos = np.array(mesh.vertices)
varr = np.hstack([uvw, pos]).flatten().astype(np.float32)
face = np.array(mesh.faces).flatten().astype(np.uint32)
shader = Shader(vs, fs)
buffer = Buffer()
frame = FrameBuffer(1024, 1024)
buffer.layout(shader, position=3, color=3)

while not glfw.window_should_close(window.window):
    w, h = glfw.get_window_size(window.window)
    frame.bind()
    buffer.bind()

    glViewport(0, 0, w, h)
    glClearColor(1.0, 0.5, 0.5, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    shader.use()
    buffer.draw(varr, face)

    glfw.poll_events()
    glfw.swap_buffers(window.window)

shader.release()
buffer.release()
glfw.terminate()


if __name__ == '__main__':
    pass
