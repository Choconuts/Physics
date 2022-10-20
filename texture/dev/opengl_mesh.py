import cv2
import numpy as np
import trimesh
from PIL import Image
from OpenGL.GL import *
import glfw
from visilab.rasterization.pipeline import MeshFilter, Buffer, Shader
from visilab.rasterization.camera import Camera
from visilab.rasterization.window import GLFWWindow
from visilab.rasterization.renderer import Renderer
from visilab.rasterization.frame import FrameBuffer


class Rasterizor:

    def __init__(self, w, h):
        self.w = w
        self.h = h

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

        self.window = GLFWWindow(1024, 1024)
        self.window.init()
        glEnable(GL_DEPTH_TEST)


        self.shader = Shader(vs, fs)
        self.buffer = Buffer()
        self.frame = FrameBuffer(w, h)
        self.buffer.layout(self.shader, position=3, color=3)

    def render(self, varr, face):
        self.frame.bind()
        self.buffer.bind()
        glViewport(0, 0, self.w, self.h)
        glClearColor(1.0, 0.5, 0.5, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        self.buffer.draw(varr, face)

        data = (GLfloat * (4 * self.w * self.h))(0)
        glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_FLOAT, data)
        image = np.frombuffer(data, np.float32, 4 * self.w * self.h).reshape([self.h, self.w, 4])
        # image = np.flip(image, -1)
        return np.flip(image, 0)

    def release(self):
        self.shader.release()
        self.buffer.release()
        glfw.terminate()


rst = Rasterizor(1024, 1024)
mesh = trimesh.load("lego_uv3.obj")
uvw = np.concatenate([mesh.visual.uv, np.zeros_like(mesh.visual.uv[..., :1])], -1)
pos = np.array(mesh.vertices)
varr = np.hstack([uvw, pos]).flatten().astype(np.float32)
face = np.array(mesh.faces).flatten().astype(np.uint32)
image = rst.render(varr, face)
cv2.imwrite("tmp.png", image * 255)
rst.release()


if __name__ == '__main__':
    pass
