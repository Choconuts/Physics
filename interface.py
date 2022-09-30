from functional import step_func, visualize_field, viewer, init_func, field, LabInstruction
viewer.auto_run = False


class UI3D(object):
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, instance, typ):
        def run(*args, **kwargs):
            gen = self.fn(instance, *args, **kwargs)
            viewer.lab_ui.pause.switch(False)

            @viewer.coroutine
            def wrap():
                yield field.ui_window()
                if gen is not None:
                    for _ in iter(gen):
                        yield field.displayables()
                while True:
                    yield field.displayables()

            viewer.run()
        return run

    def __set__(self, instance, value):
        raise NotImplementedError

    def __delete__(self, instance):
        raise NotImplementedError


def ui(f):
    return UI3D(f)


__all__ = ['ui', 'visualize_field']
