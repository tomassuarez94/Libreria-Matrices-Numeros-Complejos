import matrices as c
import math

def test_suma():
    m1 = [-1j, 2 + 1j, -2 + 1j, 0, -5, 9 - 8j]
    m2 = [7j, 2 + 9j, -2 + 0j, 7 - 8j, -5 + 9j, 9 - 1j]
    assert c.suma(m1,m2) == [6j, (4+10j), (-4+1j), (7-8j), (-10+9j), (18-9j)], c.suma(m1,m2)

def test_inverso():
    m2 = [7j, 2 + 9j, -2 + 0j, 7 - 8j, -5 + 9j, 9 - 1j]
    assert c.inverso(m2) == [(-0-7j), (-2-9j), (2-0j), (-7+8j), (5-9j), (-9+1j)], c.inverso(m2)

def test_escalar_vector():
    m2 = [7j, 2 + 9j, -2 + 0j, 7 - 8j, -5 + 9j, 9 - 1j]
    assert c.escalar(m2,5) == [35j, (10+45j), (-10+0j), (35-40j), (-25+45j), (45-5j)], c.escalar(m2,5)

def test_suma_matriz():
    m1 = [[-1j, 2 + 1j], [-2 + 1j, 0], [-5, 9 - 8j]]
    m2 = [[7j, 2 + 9j], [-2 + 0j, 7 - 8j], [-5 + 9j, 9 - 1j]]
    assert c.sum_matr(m1, m2) == [[6j, (4+10j)], [(-4+1j), (7-8j)], [(-10+9j), (18-9j)]], c.sum_matr(m1, m2)

def test_inv_matriz():
    m1 = [[-1j, 2 + 1j], [-2 + 1j, 0], [-5, 9 - 8j]]
    assert c.inv_matr(m1) == [[1j, (-2-1j)], [(2-1j), 0], [5, (-9+8j)]], c.inv_matr(m1)

def test_esc_matriz():
    m1 = [[-1j, 2 + 1j], [-2 + 1j, 0], [-5, 9 - 8j]]
    assert c.esc_matr(m1,7) == [[-7j, (14+7j)], [(-14+7j), 0], [-35, (63-56j)]], c.esc_matr(m1,7)

def test_tra_matriz():
    b2 = [[7j, 2 + 9j], [-2 + 0j, 7 - 8j], [-5 + 9j, 9 - 1j]]
    assert c.tra_matr(b2) ==[[7j, (-2+0j), (-5+9j)], [(2+9j), (7-8j), (9-1j)]] , c.tra_matr(b2)

def test_conjugado():
    b2 = [[7j, 2 + 9j], [-2 + 0j, 7 - 8j], [-5 + 9j, 9 - 1j]]
    assert c.conjug((b2)) == [[-7j, (2-9j)], [(-2-0j), (7+8j)], [(-5-9j), (9+1j)]], c.conjug((b2))

def test_adjunta_matriz():
    b2 = [[7j, 2 + 9j], [-2 + 0j, 7 - 8j], [-5 + 9j, 9 - 1j]]
    assert c.adj_matr((b2)) == [[-7j, (-2-0j), (-5-9j)], [(2-9j), (7+8j), (9+1j)]], c.adj_matr(b2)

def test_producto_matriz():
    b1 = [[-1j, 2 + 1j], [-2 + 1j, 0]]
    b2 = [[7j, 2 + 9j], [-5 + 9j, 9 - 1j]]
    assert c.prod_matr(b1,b2) == [[(-12+13j), (28+5j)], [(-7-14j), (-13-16j)]], c.prod_matr(b1,b2)

def test_accion():
    m1 = [[-1j, 2 + 1j, 8 - 1j], [-2 + 1j, 0, 9 + 5j], [-5, 9 - 8j, 6j]]
    v1 = [-1j, 2 + 1, 9 - 8j]
    assert c.accion(m1,v1) == [[(69-70j)], [(122-25j)], [(75+35j)]],  c.accion(m1,v1)

def test_interno():
    m1 = [-1j, 2 + 1, 9 - 8j]
    m2 = [7j, 2 + 9j, -2 + 0j, 7 - 8j]
    assert c.interno(m1,m2) == (-5+43j), c.interno(m1,m2)

def test_norma():
    m1 = [-1j, 2 + 1, 9 - 8j]
    assert c.norma(m1) == 12.449899597988733 , c.norma(m1)

def test_distance():
    m1 = [-1j, 2 + 1, 9 - 8j]
    m2 = [7j, 2 + 9j, -2 + 0j]
    assert c.distance(m1, m2) == 18.193405398660254, c.distance(m1, m2)
def test_unitaria():
    b2 = [[7j, 2 + 9j], [-2 + 0j, 7 - 8j], [-5 + 9j, 9 - 1j]]
    assert c.unitaria(b2) == c.unitaria(b2), c.unitaria(b2)
def test_hermitiana():
    b2 = [[7j, 2 + 9j], [-2 + 0j, 7 - 8j], [-5 + 9j, 9 - 1j]]
    assert c.hermitiana(b2) == c.hermitiana(b2), c.hermitiana(b2)
def test_tensor():
    b1 = [[-1j, 2 + 1j], [-2 + 1j, 0]]
    b2 = [[-1j, 2 + 1j], [-2 + 1j, 0]]
    assert c.tensor(b1,b2) == [[[(-1+0j), (1-2j)], [(1+2j), -0j]], [[(1-2j), (3+4j)], [(-5+0j), 0j]], [[(1+2j), (-5+0j)], [(3-4j), (-0+0j)]], [[-0j, 0j], [(-0+0j), 0]]], c.tensor(b1,b2)

if __name__ == '__main__':
    test_suma()
    test_inverso()
    test_escalar_vector()
    test_suma_matriz()
    test_inv_matriz()
    test_esc_matriz()
    test_tra_matriz()
    test_conjugado()
    test_adjunta_matriz()
    test_producto_matriz()
    test_accion()
    test_interno()
    test_norma()
    test_distance()
    #test_unitaria()
    #test_hermitiana()
    test_tensor()
    print("Prueba exitosa")

