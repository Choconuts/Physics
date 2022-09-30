from functional import step_func, visualize_field, viewer, init_func, field, LabInstruction, Inputable
import imgui
viewer.auto_run = False


class UI3D(object):
    def __init__(self, fn, opt=None):
        self.fn = fn
        self.opt = opt

    def __get__(self, instance, typ):
        def run(*args, **kwargs):
            gen = self.fn(instance, *args, **kwargs)
            viewer.lab_ui.pause.switch(False)

            @viewer.coroutine
            def wrap():
                yield field.ui_window()
                if gen is not None:
                    for _ in iter(gen):
                        if self.opt is not None:
                            yield field.displayables(), self.opt
                        else:
                            yield field.displayables()
                while True:
                    if self.opt is not None:
                        yield field.displayables(), self.opt
                    else:
                        yield field.displayables()

            viewer.run()
        return run

    def __set__(self, instance, value):
        raise NotImplementedError

    def __delete__(self, instance):
        raise NotImplementedError


def ui(opt):
    if isinstance(opt, Inputable):
        def wrap(f):
            return UI3D(f, opt)
        return wrap
    else:
        return UI3D(opt)

__all__ = ['ui', 'visualize_field', 'Inputable']
