from .. import __lib

# This file contains python wrappers for our C functions.
# The whole purpose of that is to make it easier for
# auto-completions to know our function definitions.

# __lib is the compiled library containing our c functions.


def hello():
    return __lib.hello()

def load(mx):
    return __lib.load(mx)

def eval(dt, k):
    return __lib.eval(dt, k)
