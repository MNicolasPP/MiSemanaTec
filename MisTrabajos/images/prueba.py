import cv2

image_path = "img1.jpg"  # Asegúrate de usar el nombre correcto
image = cv2.imread(image_path)

if image is None:
    print("⚠️ No se pudo cargar la imagen. Revisa la ruta y el nombre.")
else:
    print("✅ Imagen cargada correctamente.")
