from src.core.point2d import Point2D
from math import sqrt, pow


class Limb:

    def __init__(self, joint1, joint2):
        self.joint1 = joint1
        self.joint2 = joint2

    def getLength(self):
        return sqrt( pow((self.joint2.x - self.joint1.x),2) + pow((self.joint2.y - self.joint1.y),2) )

    def __str__(self):
        return "{({0},{1}),({2},{3})}".format(self.joint1.x,self.joint1.y,self.joint2.x,self.joint2.y)