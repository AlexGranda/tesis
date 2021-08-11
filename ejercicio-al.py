##python3 -m venv algebra-lineal-with-python
##. algebra-lineal-with-python/bin/activate
##pip3 install numpy
import numpy as np

#################Creando escalares
print("Creando escalares")
a = 2 ##Escalar tipo 'Int'
b = 3.1416 ##Escalar tipo 'Float'
print(a)
print(b)

#################Creando vectores
print("Creando vectores")
##1.- Convirtiendo una lista Python en un NumPy Array
a_lista = [1,2,3,4]
a_array = np.array(a_lista)
##2.- Creandolo directamente en NumPy
b_array = np.array([5,6,7])
print(a_lista)
print(a_array)


#################Creando matrices
print("Creando matrices")
print("Forma 1")
A = np.matrix([
    [3, 5],
    [1, 0]
])
print(A)

print("Forma 2")
matrix_object = np.mat([[1, 2],[1, 2],[1, 2]])
print(matrix_object)
##Tercera forma

print("Forma 3")
matrix = np.array([[1, 2],[3, 4],[5, 6]])
print(matrix)
##Podemos 'capturar' un determinado valor seleccionando su posición dentro de la matriz
print(matrix[2][0]) ##Devuelve: 5

print("")
print("")



#################Operaciones con vectores
print("Operando con vectores")
#Suma de vectores
print("Suma de vectores")
v = np.array([3, 7])
u = np.array([2, 2])
print(v + u)


##Escalar por Vector
print("Escalar por Vector")
m = np.array([4,1,7])
e = 3
mul = e * m
print(mul)


##Producto Punto
print("Producto Punto")
##Otra forma
print("Forma 1")
v = np.array([3, 7])
u = np.array([2, 2])
mul = np.dot(v,u)
print(mul)

print("Forma 2")
v = np.array([3, 7])
u = np.array([2, 2])
print(v.dot(u))


##Norma o Magnitud de un vector
print("Norma o Magnitud de un vector")
v = np.array([3, 2, 7])
print(np.linalg.norm(v))


##Vector unitario
print("Vector unitario")
def unit_vector(v):
    return v / np.linalg.norm(v)

u = np.array([3, 6, 4])
print(unit_vector(u))


##Ángulos entre vectores
print("Ángulos entre vectores")
def angle_between(v1, v2):
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(dot_pr / norms))

v = np.array([1, 4, 5])
u = np.array([2, 1, 5])
print(angle_between(v, u))



print("")
print("")


#################Operaciones con matrives

##Suma de matrices
print("Suma de matrices")
A = np.matrix([
    [3, 5],
    [1, 0]
])
B = np.matrix([
    [2, -3],
    [1, 2]
])
print(A + B)

##Escalar por matriz
print("Escalar por matriz")
A = np.matrix([
    [3, 5],
    [1, 0]
])
print(2 * A)


##Multiplicación de matrices
print("Multiplicación de matrices")
A = np.matrix([
    [3, 4],
    [1, 0]
])
B = np.matrix([
    [2, 2],
    [1, 2]
])
print(A.dot(B))


"""
Producto Hadamard
"""
print("Producto Hadamard")
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
##La multiplicación se puede realizar con multiply de NumPy y también de manera directa usando la multiplicación Python (*)
print("Forma 1")
mul = np.multiply(a,b)
##Imprimimos mul para ver la matriz respuesta
print(mul) ##Devuelve: array([[ 5, 12],[21, 32]])
print("Forma 2")
print(a*b)



print("")
print("")


#################Tipos de matrices


"""
Matriz Identidad
"""
print("Matriz Identidad")
#Declaramos la matriz identidad con identity de Numpy
print("Forma 1")
m_identidad = np.identity(5) #Devuelve la matriz identidad de tamaño 5x5
print(m_identidad) 
print("Forma 2")
A = np.eye(3)
print(A)
print("Matriz identidad de tipo float")
B = np.eye(2, dtype = float) 
print(B)

print("Especificando número de filas y columnas")
C= np.eye(4, 5) 
print(C)

#C= np.eye(4, 5, k = -1) 


"""
Matriz Transpuesta
"""
print("Matriz Transpuesta")
print("Matriz 1")
A = np.matrix([
    [3, 4],
    [1, 0]
])
print(A.T)
print("Matriz 2")
A = np.array([[1, 2], [3, 4], [5, 6]])
A_t = A.T
print(A_t)


"""
Determinante
"""
print("Determinante")
A = np.matrix([
    [3, 2],
    [1, 6]
])
print(np.linalg.det(A))


"""
Matriz Inversa
"""
print("Matriz Inversa")
#Declaramos la matriz A
A = np.array(([1,3,3],[1,4,3],[1,3,4]))
#Invertimos la matriz usando NumPy
A_inv = np.linalg.inv(A)
#Comprobamos la matriz inversa
print(A_inv)
#Devuelve la matriz inversa de A:
print("Segunda matriz inversa")
A = np.matrix([
    [4, 3],
    [5, 4]
])
print(np.linalg.inv(A))