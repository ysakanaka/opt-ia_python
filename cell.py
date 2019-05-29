#!/usr/bin/env python3


class Cell:

    def __init__(self, coord, v, a):
        self.__coordinates = coord
        self.__val = v
        self.__age = a

    @property
    def coordinates(self):
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, c):
        self.__coordinates = c

    @property
    def val(self):
        return self.__val

    @val.setter
    def val(self, v):
        self.__val = v

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, a):
        if a < 0:
            raise ValueError("Age should be over than 0.")
        self.__age = a

    def add_age(self):
        self.__age += 1

    def reset_age(self):
        self.__age = 0
