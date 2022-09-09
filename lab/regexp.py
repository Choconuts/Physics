import re
import numpy as np


def tst_case():
    ab = np.array(list("abcdefghijklmnopqrstuvwxyz.*"))
    res = np.random.choice(ab, 100)
    print(res)


def match(string, exp):
    i = 0

    for c in string:
        if i >= len(exp): return False
        if exp[i] != "*":
            if exp[i] != c and exp[i] != '.': return False
            i += 1
        else:
            if exp[i - 1] != c and exp[i - 1] != '.':
                if i < len(exp) - 1 and (exp[i + 1] == c or exp[i + 1] == '.'):
                    i += 2
                else:
                    return False
    for c in exp[i:]:
        if "*" != c: return False
    return True


def match(string, exp):
    i = 0

    for c in string:
        if i >= len(exp): return False
        if exp[i] != "*":
            if exp[i] != c and exp[i] != '.': return False
            i += 1
        else:
            if exp[i - 1] != c and exp[i - 1] != '.':
                if i < len(exp) - 1 and (exp[i + 1] == c or exp[i + 1] == '.'):
                    i += 2
                else:
                    return False
    for c in exp[i:]:
        if "*" != c: return False
    return True


if __name__ == '__main__':
    assert match("abc", "abc")
    assert not match("abc", "abcd")
    assert match("abc", "ab.")
    assert not match("aaa", "a.")
    assert match("aaa", "a.a")
    assert match("aaa", "a*")
    assert match("aab", "a*b")
    assert match("aa", ".*")
    assert match("aabb", "c*a*.b")
