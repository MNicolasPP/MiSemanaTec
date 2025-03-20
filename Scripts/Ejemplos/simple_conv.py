"""
Implementación de la detección de bordes de Sobel en Python

Autor original: Abhisek Jana
Código original: https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
Blog: http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/

Modificado por: Nicolas Peralta

Descripción:
Este código implementa la convolución de imágenes y la detección de bordes de Sobel.
Incluye funciones para aplicar un filtro de convolución a una imagen sin padding.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv_helper(fragment, kernel):
    """
    Multiplica dos matrices (subimagen y kernel) y devuelve la suma de sus productos.
    s
    Parámetros:
        fragment (numpy.ndarray): Fragmento de la imagen sobre el cual se aplica el kernel.
        kernel (numpy.ndarray): Matriz del filtro que se aplicará a la imagen.
    
    Retorna:
        float: Resultado de la suma de los productos elemento a elemento.
    """
    f_row, f_col = fragment.shape  # Dimensiones del fragmento de la imagen
    k_row, k_col = kernel.shape  # Dimensiones del kernel
    
    result = 0.0  # Inicializa el resultado de la convolución
    
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row, col] * kernel[row, col]
    
    return result

def convolution(image, kernel):
    """
    Aplica una convolución sin padding a una imagen con el kernel dado.
    
    Parámetros:
        image (numpy.ndarray): Imagen de entrada en escala de grises.
        kernel (numpy.ndarray): Matriz de filtro que se aplicará a la imagen.
    
    Retorna:
        numpy.ndarray: Imagen resultante después de aplicar la convolución.
    """
    image_row, image_col = image.shape  # Obtiene las dimensiones de la imagen
    kernel_row, kernel_col = kernel.shape  # Obtiene las dimensiones del kernel
    
    output = np.zeros(image.shape)  # Inicializa la matriz de salida con ceros
    
    # Aplicación del filtro en la imagen recorriendo cada píxel
    for row in range(image_row - kernel_row + 1):
        for col in range(image_col - kernel_col + 1):
            output[row, col] = conv_helper(
                image[row:row + kernel_row, col:col + kernel_col], kernel)
    
    # Muestra la imagen resultante en escala de grises
    plt.imshow(output, cmap='gray')
    plt.title("Imagen resultante con un kernel de {}x{}".format(kernel_row, kernel_col))
    plt.show()
    
    return output
