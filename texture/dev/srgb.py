import numpy as np


def vec3(a):
    return np.array([a, a, a])


def lessThan(a, b):
    return a < b


def mix(a, b, t):
    return a * (1 - t) + b * t


def vec4(a, b):
    return np.concatenate([a, b])


def sRGBToLinear(color):
    cutoff = lessThan(color[:3], vec3(0.04045))
    higher = np.power((color[:3] + vec3(0.055)) / vec3(1.055), vec3(2.4))
    lower = color[:3] / vec3(12.92)

    return vec4(mix(higher, lower, cutoff), color[-1:])


def linearTosRGB(color):
    cutoff = lessThan(color[:3], vec3(0.0031308))
    higher = vec3(1.055) * pow(color[:3], vec3(1.0 / 2.4)) - vec3(0.055)
    lower = color[:3] * vec3(12.92)

    return vec4(mix(higher, lower, cutoff), color[-1:])


if __name__ == '__main__':
    c1 = vec4(vec3(0.11), np.array([1.0]))
    c2 = sRGBToLinear(c1)
    print(c2)
    print(linearTosRGB(c2))
