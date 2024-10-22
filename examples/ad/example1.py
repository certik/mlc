import math

class Integer:

    def __init__(self, i):
        self.i = i

    def n(self, d):
        return self.i

    def fsdiff(self, x):
        return Integer(0)

    def ndiff(self, x, d):
        return 0

    def bndiff(self, deriv, d):
        pass

    def bclean(self):
        pass

    def bsdiff(self, deriv):
        pass

    def __str__(self):
        return f"{self.i}"

class Symbol:

    def __init__(self, name):
        self.name = name

    def n(self, d):
        return d[self]

    def fsdiff(self, x):
        if x == self:
            return Integer(1)
        else:
            return Integer(0)

    def ndiff(self, x, d):
        if x == self:
            return 1
        else:
            return 0

    def bclean(self):
        self.partial = 0

    def bndiff(self, deriv, d):
        self.partial += deriv

    # u = x
    # Input: dLdu = ∂L/∂u
    # bsdiff computes:
    # ∂L/∂x = ∂L/∂u = dLdu
    # We accumulate ∂L/∂x.
    def bsdiff(self, dLdu):
        if self.partial == 0:
            self.partial = dLdu
        else:
            self.partial = Add(self.partial, dLdu)

    def __str__(self):
        return f"{self.name}"

class Add:

    def __init__(self, a, b):
        self.left = a
        self.right = b

    def n(self, d):
        return self.left.n(d) + self.right.n(d)

    def fsdiff(self, x):
        return Add(self.left.fsdiff(x), self.right.fsdiff(x))

    def ndiff(self, x, d):
        return self.left.ndiff(x, d) + self.right.ndiff(x, d)

    def bclean(self):
        self.left.bclean()
        self.right.bclean()

    def bndiff(self, deriv, d):
        self.left.bndiff(deriv, d)
        self.right.bndiff(deriv, d)

    # u = v + w
    # Input: dLdu = ∂L/∂u
    # bsdiff computes:
    # ∂L/∂v = ∂L/∂u * ∂u/∂v = dLdu * 1 = dLdu
    # ∂L/∂w = ∂L/∂u * ∂u/∂w = dLdu * 1 = dLdu
    # We pass ∂L/∂v and ∂L/∂w down the chain.
    def bsdiff(self, dLdu):
        self.left.bsdiff(dLdu)
        self.right.bsdiff(dLdu)

    def __str__(self):
        return f"({self.left} + {self.right})"

class Mul:

    def __init__(self, a, b):
        self.left = a
        self.right = b

    def n(self, d):
        return self.left.n(d) * self.right.n(d)

    def fsdiff(self, x):
        return Add(Mul(self.left.fsdiff(x),self.right),
                Mul(self.left, self.right.fsdiff(x)))

    def ndiff(self, x, d):
        return self.left.ndiff(x,d)*self.right.n(d) + \
                self.left.n(d)*self.right.ndiff(x,d)

    def bclean(self):
        self.left.bclean()
        self.right.bclean()

    def bndiff(self, deriv, d):
        self.left.bndiff(self.right.n(d)*deriv, d)
        self.right.bndiff(self.left.n(d)*deriv, d)

    # u = v*w
    # Input: dLdu = ∂L/∂u
    # bsdiff computes:
    # ∂L/∂v = ∂L/∂u * ∂u/∂v = dLdu * w
    # ∂L/∂w = ∂L/∂u * ∂u/∂w = dLdu * v
    # We pass ∂L/∂v and ∂L/∂w down the chain.
    def bsdiff(self, dLdu):
        self.left.bsdiff(Mul(dLdu, self.right))
        self.right.bsdiff(Mul(dLdu, self.left))

    def __str__(self):
        return f"({self.left} * {self.right})"

class Sin:

    def __init__(self, a):
        self.x = a

    def n(self, d):
        return math.sin(self.x.n(d))

    def fsdiff(self, x):
        return Mul(Cos(self.x), self.x.fsdiff(x))

    def ndiff(self, x, d):
        return math.cos(self.x.n(d)) * self.x.ndiff(x,d)

    def bclean(self):
        self.x.bclean()

    def bndiff(self, deriv, d):
        self.x.bndiff(math.cos(self.x.n(d)) * deriv, d)

    # This node `Sin` represents:
    # u = sin(v)
    # where `v` can be a function of other variables (e.g., v = f(x, y, z)).

    # fsdiff: Forward mode:
    # Input: x = x
    # Output: ∂u/∂x = ∂u/∂v * ∂v/∂x = cos(v) * v.fsdiff(x)
    # The function returns the output directly, and uses `v` in the process

    # bsdiff: Reverse mode:
    # Input: dLdu = ∂L/∂u
    # bsdiff computes: ∂L/∂v = ∂L/∂u * ∂u/∂v = dLdu * cos(v)
    # The function does not return directly, rather it passes ∂L/∂v down to `v`
    # via.bsdiff(∂L/∂v). Eventually it reaches `x` (via multiple paths in
    # general) and we keep summing (accumulating) the expressions we get there.
    # The function does not call any other `bsdiff` recursively in order to
    # compute ∂L/∂v. It only calls `bsdiff` to pass the result down the tree.

    def fsdiff(self, x):
        return Mul(Cos(self.x), self.x.fsdiff(x))

    def bsdiff(u, dLdu):
        v = u.x
        dLdv = Mul(dLdu, Cos(v))
        v.bsdiff(dLdv)

    def __str__(self):
        return f"sin({self.x})"

class Cos:

    def __init__(self, a):
        self.x = a

    def n(self, d):
        return math.cos(self.x.n(d))

    def fsdiff(self, x):
        return Mul(Mul(Integer(-1), Sin(self.x)), self.x.fsdiff(x))

    def ndiff(self, x, d):
        return -math.sin(self.x.n(d)) * self.x.ndiff(x, d)

    def __str__(self):
        return f"cos({self.x})"

class Square:

    def __init__(self, a):
        self.x = a

    def n(self, d):
        return self.x.n(d)**2

    # u = v^2
    # Input: x
    # fsdiff computes:
    # ∂u/∂x = ∂u/∂v * ∂v/∂x = 2*v * v.fsdiff(x)
    # We return ∂u/∂x.
    def fsdiff(self, x):
        return Mul(Mul(Integer(2), self.x), self.x.fsdiff(x))

    def ndiff(self, x, d):
        return 2 * self.x.n(d) * self.x.ndiff(x,d)

    def bclean(self):
        self.x.bclean()

    def bndiff(self, deriv, d):
        self.x.bndiff(2*self.x.n(d)*deriv, d)

    # u = v^2
    # Input: dLdu = ∂L/∂u
    # bsdiff computes:
    # ∂L/∂v = ∂L/∂u * ∂u/∂v = dLdu * 2*v
    # We pass ∂L/∂v down the chain.
    def bsdiff(self, dLdu):
        self.x.bsdiff(Mul(dLdu, Mul(Integer(2),self.x)))

    def __str__(self):
        return f"({self.x})**2"


x = Symbol("x")
y = Symbol("y")
L = Add(Mul(x, y), Square(Sin(x)))
print("L =", L)
vals = {x: 0.1, y: 0.3}
print("L =", L.n(vals))
print("Forward:")
print("∂L/∂x =", L.fsdiff(x)) # y + 2*sin(x)*cos(x)
print("∂L/∂y =", L.fsdiff(y)) # x
print("∂L/∂x =", L.ndiff(x, vals)) # 0.49866933079506126
print("∂L/∂y =", L.ndiff(y, vals)) # 0.1
print("Backward:")
L.bclean()
L.bsdiff(Integer(1))
print("∂L/∂x =", x.partial) # y + 2*sin(x)*cos(x)
print("∂L/∂y =", y.partial) # x
L.bclean()
L.bndiff(1, vals)
print("∂L/∂x =", x.partial) # 0.49866933079506126
print("∂L/∂y =", y.partial) # 0.1
print()
print()

# https://en.wikipedia.org/wiki/Automatic_differentiation
# Example: Finding the partials of z = x * (x + y) + y * y at (x, y) = (2, 3)
z = Add(Mul(x, Add(x, y)), Mul(y, y))
vals = {x: 2, y: 3}

print("z =", z.n(vals))             # Output: z = 19
print("Forward:")
print("∂z/∂x =", z.fsdiff(x))  # Output: 2*x + y
print("∂z/∂y =", z.fsdiff(y))  # Output: x + 2*y
print("∂z/∂x =", z.ndiff(x, vals))  # Output: ∂z/∂x = 7
print("∂z/∂y =", z.ndiff(y, vals))  # Output: ∂z/∂y = 8
print("Backward:")
z.bclean()
z.bndiff(1, vals)
print("∂z/∂x =", x.partial)  # Output: ∂z/∂x = 7
print("∂z/∂y =", y.partial)  # Output: ∂z/∂y = 8
z.bclean()
z.bsdiff(Integer(1))
print("∂z/∂x =", x.partial)  # Output: 2*x + y
print("∂z/∂y =", y.partial)  # Output: x + 2*y
print()
print()


################################################################################

L = Mul(Mul(x, y), Square(Square(Sin(x))))
print("L =", L)
print("Forward:")
print("∂L/∂x =", L.fsdiff(x))
print("∂L/∂y =", L.fsdiff(y))
print("Backward:")
L.bclean()
L.bsdiff(Integer(1))
print("∂L/∂x =", x.partial)
print("∂L/∂y =", y.partial)
