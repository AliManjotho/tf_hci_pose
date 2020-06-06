class Point2D:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __sub__(self, other):
        p = Point2D()
        p.x = other.x - self.x
        p.y = other.y - self.y
        return p

    def __add__(self, other):
        p = Point2D()
        p.x = self.x + other.x
        p.y = self.y + other.y
        return p

    def cross(self, other):
        return (self.x * other.y) - (self.y * other.x)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y)

    def __str__(self):
        return "({0},{1})".format(self.x, self.y)