from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from visilab.render_api import *
import numpy as np
import os


class FrameBuffer:

    def __init__(self, w, h, data_type=GL_UNSIGNED_BYTE):
        self.fbo = glGenFramebuffers(1)
        self.tcb = glGenTextures(1)
        self.rbo = glGenRenderbuffers(1)
        self.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, data_type, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tcb, 0)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("[ERROR]", "Frame buffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glBindTexture(GL_TEXTURE_2D, self.tcb)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

    def release(self):
        self.bind()
        glDeleteBuffers(1, GL_FRAMEBUFFER)
        glDeleteTextures(1, GL_TEXTURE_2D)
        glDeleteRenderbuffers(1, GL_RENDERBUFFER)
        self.unbind()
