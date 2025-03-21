import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_padding(image, kernel_shape, extra_padding=0):
    """Agrega padding a la imagen basado en el tamaño del kernel y padding adicional."""
    kernel_row, kernel_col = kernel_shape
    pad_height = int((kernel_row - 1) / 2) + extra_padding
    pad_width = int((kernel_col - 1) / 2) + extra_padding
    
    padded_image = np.zeros((
        image.shape[0] + 2 * pad_height,
        image.shape[1] + 2 * pad_width
    ))
    
    padded_image[pad_height:pad_height + image.shape[0], pad_width:pad_width + image.shape[1]] = image
    return padded_image

def convolution(image, kernel, average=False, extra_padding=0, verbose=False):
    """Aplica convolución a una imagen con un kernel específico."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if verbose:
            print("Convertida a escala de grises.")

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros((image_row, image_col))

    padded_image = apply_padding(image, kernel.shape, extra_padding)

    for row in range(image_row):
        for col in range(image_col):
            region = padded_image[row:row + kernel_row, col:col + kernel_col]
            output[row, col] = np.sum(kernel * region)
            if average:
                output[row, col] /= kernel.size

    return output

def main():
    # Cargar imagen (puedes cambiar esto a tu ruta)
    image_path = "images/img1.jpg"  # O "img2.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("No se pudo cargar la imagen. Verifica el nombre o ruta.")
        return

    # Mostrar imagen original
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Imagen original")
    plt.axis('off')
    plt.show()

    # Kernel de ejemplo (detector de bordes Sobel en X)
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Aplicar convolución
    resultado = convolution(image, kernel, average=False, extra_padding=0, verbose=False)

    # Mostrar imagen con filtro
    plt.imshow(resultado, cmap='gray')
    plt.title("Imagen con convolución aplicada")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
