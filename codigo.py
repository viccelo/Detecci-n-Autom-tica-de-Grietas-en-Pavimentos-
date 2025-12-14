import os
import shutil
import random

# Ruta base del proyecto
base_dir = "P"

# Rutas de las carpetas originales
cd_dir = os.path.join(base_dir, "CP")  # Con grietas
ud_dir = os.path.join(base_dir, "UP")  # Sin grietas

# Carpetas destino
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Porcentaje de entrenamiento
train_split = 0.8  # 80% entrenamiento, 20% prueba

# Crear carpetas si no existen
for folder in [train_dir, test_dir]:
    for category in ["CP", "UP"]:
        os.makedirs(os.path.join(folder, category), exist_ok=True)

def dividir_imagenes(src_folder, dest_train, dest_test, split_ratio):
    """Divide imágenes aleatoriamente en entrenamiento y prueba."""
    images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(images)

    # Calcular cuántas serán para entrenamiento
    train_count = int(len(images) * split_ratio)

    train_images = images[:train_count]
    test_images = images[train_count:]

    # Copiar archivos a las carpetas correspondientes
    for img in train_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(dest_train, img))

    for img in test_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(dest_test, img))

    print(f"{os.path.basename(src_folder)}: {len(train_images)} entrenamiento, {len(test_images)} prueba")

# Semilla fija (opcional, para que siempre se divida igual)
random.seed(42)

# Dividir las carpetas CP y UP
dividir_imagenes(cd_dir,
                 os.path.join(train_dir, "CP"),
                 os.path.join(test_dir, "CP"),
                 train_split)

dividir_imagenes(ud_dir,
                 os.path.join(train_dir, "UP"),
                 os.path.join(test_dir, "UP"),
                 train_split)

print("✅ División completada con éxito.")