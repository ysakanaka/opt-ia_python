#!/usr/bin/env python3

import numpy as np


class Cell:

    def __init__(self, coord, v, a):
        self.coordinates = coord
        self.val = v
        self.age = a

    def set_coordinates(self, c):
        self.coordinates = c

    def set_val(self, v):
        self.val = v

    def set_age(self, a):
        self.age = a

    def get_coordinates(self):
        return self.coordinates

    def get_array_coordinates(self):
        return [list(np.array(self.coordinates).copy())]

    def get_val(self):
        return self.val

    def get_age(self):
        return self.age

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0
