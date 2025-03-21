# Explicacion

## Función apply_padding()
Calcula cuánto padding agregar en altura (pad_height) y en ancho (pad_width).
Crea una nueva imagen más grande llena de ceros (np.zeros), lo que significa que el padding será negro.
Inserta la imagen original en el centro de la nueva imagen con padding.
Devuelve la imagen con padding aplicado.

### 🖼️ Ejemplo Visual de Cómo Funciona
Supongamos que la imagen original es:

| 1 | 2 | 3 |
|-------|-------|-------|
| 4  | 5   | 6   |
| 7  | 8   | 9   |

Si agregamos padding = 1, la imagen se ve así:
| 0 | 0 | 0 | 0 | 0 |
|-------|--|-----|---|----|
| 0 |1 | 2 | 3 | 0 |
| 0 | 4  | 5   | 6   | 0 |
|  0 |7  | 8   | 9   | 0 |
| 0 | 0 | 0 | 0 | 0 |

Luego, la convolución recorre la imagen ventana por ventana aplicando la multiplicación del kernel y sumando los valores.



## Función convolution()
Si la imagen tiene 3 canales (RGB), la convierte a escala de grises (cv2.COLOR_BGR2GRAY).
Obtiene el tamaño de la imagen y el del kernel.
Aplica padding a la imagen usando apply_padding().
Crea una imagen vacía output donde se guardarán los valores de la convolución.

Bucle de convolución:
- Recorre la imagen pixel por pixel.
- Extrae una región del mismo tamaño que el kernel.
- Multiplica la región con el kernel y suma los valores (np.sum()).
- Si average=True, divide el resultado entre el tamaño del kernel para hacer un suavizado.

#### 📌 ¿Qué es la convolución? 
Es una operación matemática donde aplicamos un filtro (kernel) sobre una imagen para resaltar características como bordes, desenfoque o nitidez.

