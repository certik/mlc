import math

class Integer:

    def __init__(self, i):
        self.i = i

    def n(self, d):
        return self.i

    def sdiff(self, x):
        return Integer(0)

    def __str__(self):
        return f"{self.i}"

class Symbol:

    def __init__(self, name):
        self.name = name

    def n(self, d):
        return d[self]

    def sdiff(self, x):
        if x == self:
            return Integer(1)
        else:
            return Integer(0)

    def __str__(self):
        return f"{self.name}"

class Add:

    def __init__(self, a, b):
        self.left = a
        self.right = b

    def n(self, d):
        return self.left.n(d) + self.right.n(d)

    def sdiff(self, x):
        return Add(self.left.sdiff(x), self.right.sdiff(x))

    def __str__(self):
        return f"({self.left} + {self.right})"

class Mul:

    def __init__(self, a, b):
        self.left = a
        self.right = b

    def n(self, d):
        return self.left.n(d) * self.right.n(d)

    def sdiff(self, x):
        return Add(Mul(self.left.sdiff(x),self.right),
                Mul(self.left, self.right.sdiff(x)))

    def __str__(self):
        return f"({self.left} * {self.right})"

class Sin:

    def __init__(self, a):
        self.x = a

    def n(self, d):
        return math.sin(self.x.n(d))

    def sdiff(self, x):
        return Mul(Cos(self.x), self.x.sdiff(x))

    def __str__(self):
        return f"sin({self.x})"

class Cos:

    def __init__(self, a):
        self.x = a

    def n(self, d):
        return math.cos(self.x.n(d))

    def sdiff(self, x):
        return Mul(Mul(Integer(-1), Sin(self.x)), self.x.sdiff(x))

    def __str__(self):
        return f"cos({self.x})"

class Square:

    def __init__(self, a):
        self.x = a

    def n(self, d):
        return self.x.n(d)**2

    def sdiff(self, x):
        return Mul(Mul(Integer(2), self.x), self.x.sdiff(x))

    def __str__(self):
        return f"({self.x})^2"


x = Symbol("x")
y = Symbol("y")
L = Add(Mul(x, y), Square(Sin(x)))
print(L)
print(L.n({x: 0.1, y: 0.3}))
print(L.sdiff(x))
print(L.sdiff(y))
