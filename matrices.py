import math
import numpy
def suma(a,b):
    if len(a) == len(b):
        c = []
        for i in range(len(a)):
            c.append(a[i] + b[i])
        return c
    else:
        raise Exception("Error")
def inverso(a):
    c = []
    for i in range(len(a)):
        c.append(-a[i])
    return c
def escalar(a,k):
    c = []
    for i in range(len(a)):
        c.append(k * a[i])
    return c
def sum_matr(a,b):
    if len(a) == len(b) and len(a[0]) == len(b[0]):
        c = []
        for i in range(len(a)):
            c.append([])
            for j in range(len(a[0])):
                c[i].append(0)
        for i in range(len(a)):
            for j in range(len(a[0])):
                c[i][j]= a[i][j]+ b[i][j]
        return c
    else:
        raise Exception ("Error")
def inv_matr(a):
    c = []
    for i in range(len(a)):
        c.append([])
        for j in range(len(a[0])):
            c[i].append(0)
    for i in range(len(a)):
        for j in range(len(a[0])):
            c[i][j]= -1*a[i][j]
    return c
def esc_matr(a,k):
    c = []
    for i in range(len(a)):
        c.append([])
        for j in range(len(a[0])):
            c[i].append(0)
    for i in range(len(a)):
        for j in range(len(a[0])):
            c[i][j]= k*a[i][j]
    return c
def tra_matr(a):
    c = [[0 for i in range(len(a))] for j in range(len(a[0]))]
    for i in range(len(a[0])):
        for j in range(len(a)):
            c[i][j] = a[j][i]
    return c
def conjug(a):
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j] = a[i][j].conjugate()
    return a
def adj_matr(a):
    return conjug(tra_matr(a))
def prod_matr(a, b):
    if len(a[0]) != len(b):
        raise Exception("Error")
    producto = []
    for i in range(len(b)):
        producto.append([])
        for j in range(len(b[0])):
            producto[i].append(None)
    for c in range(len(b[0])):
        for i in range(len(a)):
            suma = 0
            for j in range(len(a[0])):
                suma += a[i][j]*b[j][c]
            producto[i][c] = suma
    return producto
def adj_vector(v):
    for i in range(len(v)):
        v[i]= v[i].conjugate()
    return v
def cambio(vector):
    vector1 = [[(0,0) for j in range(1)] for i in range(len(vector))]
    for i in range(len(vector)):
        vector1[i][0] = vector[i]
    return  vector1
def accion(mat, vector):
    matR = prod_matr(mat, cambio(vector))
    return matR
def interno(vector, vector2):
    if len(vector) != len(vector2):
        raise Exception ("ERROR")
    vector = adj_vector(vector)
    res = 0
    for i in range(len(vector)):
        res += vector[i]*vector2[i]
    return res
def norma(vec):
    a = []
    for i in range(len(vec)):
        a.append(vec[i])
    return math.sqrt(interno(vec,a).real)
def distance(a,b):
    if len(a) == len(b):
        return math.sqrt(math.fabs(norma(a)-norma(b)))
    else:
        raise Exception("Error")
def propios(mat):
    return numpy.linalg.eigh(mat)
def unitaria(m):
    if len(m) != len(m[0]):
        raise Exception("No unitaria")
    u = numpy.identity(len(m))
    x = (prod_matr(m,tra_matr(m)))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] != u[i][j]:
                raise Exception("No unitaria")
    return "Unitaria"
def hermitiana(m):
    if len(m) != len(m[0]):
        raise Exception("No hermitiana")
    x = adj_matr(m)
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] != x[i][j]:
                raise Exception("No Hermitiana")
    return "Hermitiana"
def tensor(a,b):
    c = [[0 for i in range(len(a)*len(b))] for j in range(len(a[0])*len(b[0]))]
    for i in range(len(b)):
        for j in range(len(b[0])):
            c[i][j] = a[i//len(b)][j//len(b[0])] * b[i % len(b)][j % len(b[0])]
    return c
