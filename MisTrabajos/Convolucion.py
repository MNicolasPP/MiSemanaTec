import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_padding(image, kernel_shape, extra_padding=5):
    """Agrega padding a la imagen basado en el tamaño del kernel y un padding adicional."""
    kernel_row, kernel_col = kernel_shape
    pad_height = int((kernel_row - 1) / 2) + extra_padding
    pad_width = int((kernel_col - 1) / 2) + extra_padding
    
    padded_image = np.zeros((
        image.shape[0] + 2 * pad_height,
        image.shape[1] + 2 * pad_width
    ), dtype=np.uint8)
    
    padded_image[pad_height:pad_height + image.shape[0], pad_width:pad_width + image.shape[1]] = image
    return padded_image

def convolution(image, kernel, average=False, extra_padding=20):
    """Aplica convolución a una imagen con un kernel específico."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Aplicar padding
    padded_image = apply_padding(image, kernel.shape, extra_padding)

    # Crear imagen de salida
    output = np.zeros((image_row, image_col), dtype=np.float32)

    for row in range(image_row):
        for col in range(image_col):
            region = padded_image[row:row + kernel_row, col:col + kernel_col]
            output[row, col] = np.sum(kernel * region)
            if average:
                output[row, col] /= kernel.size

    return padded_image, output

def main():
    # Ruta de la imagen (asegúrate de que el archivo existe)
    image_path = "images/img1.jpg"  # Cambia a "img2.jpg" si quieres probar con otra imagen
    image = cv2.imread(image_path)

    if image is None:
        print("No se pudo cargar la imagen")
        return

    # Mostrar imagen original
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Original")
    plt.axis('off')

    # Kernel para detección de bordes (Sobel en X)
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Aplicar convolución y obtener imagen con padding
    padded_image, convoluted_image = convolution(image, kernel, average=False, extra_padding=5)

    # Mostrar imagen con padding
    plt.subplot(1, 3, 2)
    plt.imshow(padded_image, cmap='gray')
    plt.title("Imagen con Padding")
    plt.axis('off')

    # Mostrar imagen después de convolución
    plt.subplot(1, 3, 3)
    plt.imshow(convoluted_image, cmap='gray')
    plt.title("Imagen con Convolucion")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
