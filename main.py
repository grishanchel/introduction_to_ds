# import numpy as np
# import typing as tp
#
# def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
#
#     A_1 = np.expand_dims(A, axis=2)
#     B_1 = np.expand_dims(B, axis=0)
#
#     result = np.sum(A_1 * B_1, axis=1)
#     return result
#
#
# # Пример использования
# A = np.array([[1, 2], [3,4]])
# B = np.array([[2, 0], [1, 2]])
# print(matrix_multiplication(A, B))
#
#
# x = np.array([[1, 2, 3], [2, 3, 4]])
# print(x.shape)
#
# y = np.expand_dims(x, axis=2)
# print(y)
# print(y.shape)
# #
# # y = np.expand_dims(x, axis=0)
# # print(y)
# # print(y.shape)


import numpy as np

# # Вычисление евклидова расстояния между двумя точками
# def euclidean_distance(point1, point2):
#     return np.sqrt(np.sum((point1 - point2) ** 2))
#
# # Нахождение k ближайших соседей
# def find_nearest_points(train_set, test_instance, k):
#     distances = np.apply_along_axis(lambda x: euclidean_distance(test_instance[:-1], x[:-1]), 1, train_set)
#     nearest_indices = np.argpartition(distances, k)[:k]
#     return train_set[nearest_indices+1].T


# def solution(A, B, k):
#     distances = np.linalg.norm((A[:, None] - B), axis=-1)
#
#     nearest_indices = np.argsort(distances, axis=0)[:k,:]
#
#     return (nearest_indices+1).T

import numpy as np
import typing as tp

def matrix_multiplication(A, B):

    A_1 = np.expand_dims(A, axis=2)
    B_1 = np.expand_dims(B, axis=0)

    result = np.sum(A_1 * B_1, axis=1)
    return result
def find_nearest_points(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    A = np.expand_dims(A, axis=0)
    B = np.expand_dims(B, axis=0)

    A_squared = np.sum(A**2, axis=1, keepdims=True)
    B_squared = np.sum(B**2, axis=1, keepdims=True)
    AB = matrix_multiplication(A, B.T)

    distances = np.sqrt(A_squared - 2 * AB + B_squared.T)

    nearest_indices = np.argsort(distances, axis=1)[:,:k]

    return nearest_indices

# Пример использования
A = np.array([[1]])
B = np.array([[3]])
k = 1
C = find_nearest_points(A, B, k)
print(C)

A = np.array([
[0, 0],
[1, 0],
[2, 0]])

B = np.array([
[0, 1],
[2, 1]])

print(find_nearest_points(A, B, 2))















