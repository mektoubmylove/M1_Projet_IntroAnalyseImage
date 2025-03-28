import json
import math
import cv2
import numpy as np
import os
import hashlib
import random
from collections import defaultdict

def load_image(path):
    """
    Load an image from the specified path using OpenCV.

    Parameters:
        path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image.
    """
    return cv2.imread(path)

def resize_image(image, height):
    """
    Resize the input image to a specified height while maintaining aspect ratio.

    Parameters:
        image (numpy.ndarray): The input image.
        height (int): The desired height of the resized image.

    Returns:
        numpy.ndarray: The resized image.
    """
    return cv2.resize(image, (height, int(height * image.shape[0] / image.shape[1])))

def average_brightness(hsv_image):
    """
    Calculate the average brightness of an HSV (hue, saturation, value) image.

    Parameters:
        hsv_image (numpy.ndarray): The input image in HSV color space.

    Returns:
        float: The average brightness value.
    """
    return np.mean(hsv_image[:, :, 2])

def brighten_image(hsv_image, lower_threshold):
    """
    Brighten the input HSV image if its average brightness is below a specified threshold.

    Parameters:
        hsv_image (numpy.ndarray): The input image in HSV color space.
        lower_threshold (float): The threshold below which the image will be brightened.

    Returns:
        numpy.ndarray: The brightened image in HSV color space.
    """
    image = hsv_image.copy()
    image_average_brightness = average_brightness(image)
    print(f"Mean = {image_average_brightness}")

    if image_average_brightness != lower_threshold:
        enhanced_brightness = 180  # Fixed brightness value to enhance to
        image[:, :, 2] = np.clip(image[:, :, 2] * (enhanced_brightness / image_average_brightness), 0, 255)

    return image

def hsv_image(image):
    """
    Convert an input BGR image to HSV (hue, saturation, value) color space.

    Parameters:
        image (numpy.ndarray): The input BGR image.

    Returns:
        numpy.ndarray: The image converted to HSV color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def gray_scale(image):
    """
    Convert an input BGR image to grayscale.

    Parameters:
        image (numpy.ndarray): The input BGR image.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("uint8")

def threshold(image, low_t, high_t):
    image = cv2.imread(image)

    assert low_t <= high_t

    # Conversion en niveaux de gris si l'image est en couleur
    if len(image.shape) == 3:  # Image couleur
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = img.copy()

    # Dimensions de l'image
    height, width = img.shape


    # Appliquer le seuillage
    for i in range(height):
        for j in range(width):
            if img[i, j] < low_t:
                res[i, j] = 0
            elif img[i, j] > high_t:
                res[i, j] = 255

    return res

def normalize(image,minValue,maxValue):
    image=cv2.imread(image)

    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin du fichier.")
    assert (minValue <= maxValue)

    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = img.copy()

    # Dimensions de l'image
    height, width = res.shape

    #trouver fmin fmax
    fmax = 0
    fmin = 255
    for i in range(height):
        for j in range(width):
            if img[i][j] > fmax:
                fmax = img[i][j]
            if img[i][j] < fmin:
                fmin = img[i][j]

    """
    fmin = np.min(img)
    fmax = np.max(img)
    """

    """
    for i in range(height):
        for j in range(width):
            res[i][j]=(img[i][j]- fmin)*((maxValue-minValue)/(fmax-fmin))+minValue
    """
    res = (img - fmin) * ((maxValue - minValue) / (fmax - fmin)) + minValue
    res = res.astype(np.uint8)

    return res
def quantize(image,numberOfLevels):
    image = cv2.imread(image)

    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin du fichier.")

    if numberOfLevels <= 0:
        raise ValueError("Le nombre de niveaux doit être positif.")

    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    img = img.astype(np.float32)/255.0


    interval_size = 1.0 / numberOfLevels


    quantized_image = np.zeros_like(img)


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            level = int(img[i, j] // interval_size)  # Trouver l'intervalle du pixel
            quantized_value = (level+0.5 ) * interval_size
            quantized_image[i, j] = quantized_value


    quantized_image = (quantized_image*255 ).astype(np.uint8)


    img = (img*255 ).astype(np.uint8)
    return quantized_image
def equalize(image):
    img=cv2.imread(image)

    if img is None:
        raise ValueError("chemin de l'image incorrect")

    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res=np.zeros_like(image, dtype=np.float32)

    hist=[0]*256
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            valeur=int(image[i][j])
            hist[valeur]+=1

    histCumule=[0]*256
    histCumule[0]=hist[0]
    for i in range(1,256):
        histCumule[i]=histCumule[i-1]+hist[i]

    taille=image.shape[0]*image.shape[1]

    histCumuleNorm=[0]*256
    for i in range(256):
        histCumuleNorm[i]=histCumule[i]/taille

    vmin = np.min(img)
    vmax = np.max(img)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res[i][j]= (vmax-vmin)*histCumuleNorm[int(image[i,j])]+vmin

    res=np.uint8(res)
    return res
def color2gray(image):
    image=cv2.imread(image)

    gray_img=np.zeros(image.shape,dtype=np.uint8)

    height,width=image.shape[:2]

    for i in range(height):
        for j in range(width):
            gray_img[i][j]=(int(image[i][j][0])+int(image[i][j][1])+int(image[i][j][2]))//3
    return gray_img

def tresholdOtsu(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin du fichier.")

    # Dimensions de l'image
    height, width = image.shape
    total_pixels = height * width

    hist = [0] * 256
    for i in range(height):
        for j in range(width):
            intensity = int(image[i, j])
            hist[intensity] += 1

    prob = [0] * 256  # Initialisation d'une liste de probabilités
    for i in range(256):
        prob[i] = hist[i] / total_pixels

    best_threshold = 0
    max_variance = 0

    # Parcourir tous les seuils possibles
    for t in range(256):
        # Calculer les poids des deux classes
        w0 = sum(prob[:t])
        w1 = sum(prob[t:])

        # Éviter les divisions par zéro
        if w0 == 0 or w1 == 0:
            continue

        # Calculer les moyennes des deux classes
        mean0 = sum(i * prob[i] for i in range(t)) / w0
        mean1 = sum(i * prob[i] for i in range(t, 256)) / w1

        # Calculer la variance inter-classe
        variance = w0 * w1 * (mean0 - mean1) ** 2

        # Mettre à jour le meilleur seuil si nécessaire
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    binary_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] >= best_threshold:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0
    return binary_image,best_threshold

def meanFilter(image,k):
    image=cv2.imread(image)
    if len(image.shape)==3:
        img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        img=image
    height,width=img.shape

    padded_img = np.zeros((height + 2 * k, width + 2 * k), dtype=np.uint8)


    # Copier l’image originale au centre du padding
    for i in range(height):
        for j in range(width):
            padded_img[i + k][j + k] = img[i, j]

    filtered_image = np.zeros((height, width), dtype=np.uint8)

    # Appliquer le filtre moyenneur
    for i in range(height):
        for j in range(width):
            # Somme des intensités dans la fenêtre
            sum_pixels = 0
            for m in range(-k, k + 1):
                for n in range(-k, k + 1):
                    sum_pixels += padded_img[i + k + m][j + k + n]

            # Calculer la moyenne
            filtered_image[i, j] = sum_pixels // ((2 * k + 1) ** 2)
    return filtered_image

def convolution(image,kernel):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin du fichier.")

    height, width = image.shape
    k_height, k_width = kernel.shape  # Dimensions du noyau
    pad_h, pad_w = k_height // 2, k_width // 2  # Padding nécessaire

    # Ajouter un padding (zéro-padding)
    padded_img = np.zeros((height + 2 * pad_h, width + 2 * pad_w), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            padded_img[i + pad_h, j + pad_w] = image[i, j]

    # Initialiser l'image résultante
    output_image = np.zeros((height, width), dtype=np.float32)

    # Appliquer la convolution
    for i in range(height):
        for j in range(width):
            sum_pixels = 0  # Stocker la somme pondérée
            for m in range(k_height):
                for n in range(k_width):
                    sum_pixels += padded_img[i + m, j + n] * kernel[m, n]
            output_image[i, j] = sum_pixels  # Mettre la valeur convoluée dans l'image finale

    # Normaliser l'image pour s'assurer que les valeurs sont entre 0 et 255
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return output_image

def edgeSobel(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin du fichier.")

    # Dimensions de l'image
    height, width = image.shape

    # Définition des kernels de Sobel
    Gx_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # Ajouter un padding de 1 (zero-padding) autour de l'image
    padded_img = np.zeros((height + 2, width + 2), dtype=np.uint8)
    padded_img[1:height + 1, 1:width + 1] = image

    # Créer une image vide pour stocker la norme du gradient
    gradient_magnitude = np.zeros((height, width), dtype=np.uint8)

    # Appliquer la convolution avec les kernels de Sobel
    for i in range(1, height + 1):  # Parcours de l'image sans le padding
        for j in range(1, width + 1):
            Gx = 0
            Gy = 0

            # Appliquer la convolution avec les matrices de Sobel
            for m in range(3):  # Kernel 3x3
                for n in range(3):
                    Gx += Gx_kernel[m][n] * padded_img[i + m - 1][j + n - 1]
                    Gy += Gy_kernel[m][n] * padded_img[i + m - 1][j + n - 1]

            # Calculer la norme du gradient
            G = np.sqrt(Gx ** 2 + Gy ** 2)

            # Normaliser la valeur (entre 0 et 255)
            gradient_magnitude[i - 1, j - 1] = min(255, int(G))
    return gradient_magnitude
def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))

def bilateralFilter(image, kernel_size, sigma_r):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin du fichier.")
    # Assurez-vous que le noyau a une taille impaire
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Créez une image de sortie vide
    output = np.zeros_like(image)

    # Calculer la moitié de la taille du noyau
    half_kernel = kernel_size // 2

    # Parcourir chaque pixel de l'image
    for i in range(half_kernel, image.shape[0] - half_kernel):
        for j in range(half_kernel, image.shape[1] - half_kernel):
            # Initialiser les valeurs pour le pixel courant
            total_weight = 0.0
            filtered_value = 0.0

            # Parcourir le voisinage du pixel
            for x in range(-half_kernel, half_kernel + 1):
                for y in range(-half_kernel, half_kernel + 1):
                    # Calculer la distance spatiale
                    spatial_distance = math.sqrt(x**2 + y**2)
                    spatial_weight = gaussian(spatial_distance, sigma_r)

                    # Calculer la différence d'intensité
                    intensity_difference = image[i, j] - image[i + x, j + y]
                    range_weight = gaussian(intensity_difference, sigma_r)

                    # Calculer le poids total
                    weight = spatial_weight * range_weight

                    # Accumuler les valeurs pondérées
                    filtered_value += image[i + x, j + y] * weight
                    total_weight += weight

            # Normaliser la valeur filtrée
            output[i, j] = filtered_value / total_weight
    return output

def median(image,size):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin.")

    height, width = image.shape
    k = size // 2  # Taille du décalage pour la fenêtre

    # Ajouter un zero-padding
    padded_img = np.zeros((height + 2 * k, width + 2 * k), dtype=np.uint8)
    padded_img[k:-k, k:-k] = image

    # Image de sortie
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    # Appliquer le filtre médian
    for i in range(height):
        for j in range(width):
            # Extraire la fenêtre
            window = []
            for m in range(-k, k + 1):
                for n in range(-k, k + 1):
                    window.append(padded_img[i + k + m, j + k + n])

            # Calculer la médiane
            window.sort()
            median_value = window[len(window) // 2]

            # Affecter la valeur médiane
            filtered_image[i, j] = median_value
    return filtered_image

def parcoursCC(image, p, label, labels_image):
    """
    Parcours en profondeur pour l'étiquetage des composantes connexes.
    """
    # Initialisation de la pile
    stack = [p]
    # Marquer le pixel comme visité en l'étiquetant
    labels_image[p[1], p[0]] = label
    size=1 #nombre de pixels de la composante

    # Directions pour l'adjacence 4 (haut, bas, gauche, droite)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Tant que la pile n'est pas vide, on continue l'exploration
    while stack:
        current_pixel = stack.pop()

        # Parcourir les voisins du pixel courant
        for direction in directions:
            voisin_x = current_pixel[0] + direction[0]
            voisin_y = current_pixel[1] + direction[1]

            # Vérifier si le voisin est dans l'image et si c'est un pixel non visité et un pixel de valeur 1
            if (0 <= voisin_x < image.shape[1] and 0 <= voisin_y < image.shape[0]
                    and image[voisin_y, voisin_x] == 1 and labels_image[voisin_y, voisin_x] == 0):
                # Marquer ce pixel comme visité et l'ajouter à la pile
                labels_image[voisin_y, voisin_x] = label
                stack.append((voisin_x, voisin_y))
                size+=1
    return size

def ccLabel(image_path):
    """
    Fonction d'étiquetage des composantes connexes dans une image binaire.
    Chaque composante connexe reçoit une couleur aléatoire.
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Vérifier si l'image est bien chargée
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin.")

    # Binariser l'image (128 -> valeur arbitraire)
    image = np.where(image > 128, 1, 0).astype(np.uint8)

    # Créer une image pour les labels
    labels_image = np.zeros_like(image, dtype=np.int32)

    # Créer une image en couleur pour afficher les résultats
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)


    label = 1  # Compteur pour identifier chaque composante connexe

    # Parcourir tous les pixels de l'image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Si le pixel est un pixel blanc et n'a pas été étiqueté
            if image[y, x] == 1 and labels_image[y, x] == 0:
                # Appeler la fonction parcoursCC pour marquer tous les pixels connectés
                parcoursCC(image, (x, y), label, labels_image)

                # on assigne une couleur aléatoire à cette composante connexe
                color = [random.randint(0, 255) for c in range(3)]

                # Colorer les pixels de la composante connexe
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        if labels_image[i, j] == label:
                            colored_image[i, j] = color

                label += 1  # Incrémenter le label pour la prochaine composante connexe
    return colored_image

def ccAreaFilter(image_path, size):
    """
    Filtre d’aire : supprime les composantes connexes de taille < size.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin.")

    binary_image = np.where(image > 128, 1, 0).astype(np.uint8)

    #image pour stocker les etiquettes des composantes
    labels_image = np.zeros_like(binary_image, dtype=np.int32)
    taille_composantes = {}  # Dictionnaire pour stocker la taille des composantes

    label = 1 # Compteur pour identifier chaque composante connexe

    # Parcourir tous les pixels de l'image
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 1 and labels_image[y, x] == 0:
                #trouver la composante et recup sa taille
                taille_compo = parcoursCC(binary_image, (x, y), label, labels_image)

                # Enregistrer la taille de la composante
                if taille_compo is not None:
                    taille_composantes[label] = taille_compo
                    label += 1 #on passe au label suivant

    # Image filtrée où seules les composantes de taille suffisante seront conservées
    filtered_image = np.zeros_like(binary_image, dtype=np.uint8)

    #parcourir l'image avec les etuqttes pour filtrer les composantes < size
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            label_value = labels_image[y, x]

            # le label est valide et taille est suffisante
            if label_value > 0 and label_value in taille_composantes:
                if taille_composantes[label_value] >= size:
                    filtered_image[y, x] = 255  # Conserver le pixel dans l'image filtrée
    return filtered_image

def apply_gaussian_kernel(image, kernel):
    """
    Apply a Gaussian kernel to the input image.

    Parameters:
        image (numpy.ndarray): The input grayscale image.
        kernel (numpy.ndarray): The Gaussian kernel.

    Returns:
        numpy.ndarray: The image after applying the Gaussian blur.
    """
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad_width = kernel_size // 2

    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    output_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output_image[i, j] = np.sum(region * kernel)
    
    return output_image

def gaussian_blur(image):
    """
    Apply Gaussian blur to the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image after applying Gaussian blur.
    """
    kernel = np.array([[0.00097656, 0.00341797, 0.00683594, 0.00878906, 0.00683594,
  0.00341797, 0.00097656],
  [0.00341797, 0.01196289, 0.02392578, 0.03076172, 0.02392578,
  0.01196289, 0.00341797],
  [0.00683594, 0.02392578, 0.04785156, 0.06152344, 0.04785156,
  0.02392578, 0.00683594],
  [0.00878906, 0.03076172, 0.06152344, 0.07910156, 0.06152344,
  0.03076172, 0.00878906],
  [0.00683594, 0.02392578, 0.04785156, 0.06152344, 0.04785156,
  0.02392578, 0.00683594],
  [0.00341797, 0.01196289, 0.02392578, 0.03076172, 0.02392578,
  0.01196289, 0.00341797],
  [0.00097656, 0.00341797, 0.00683594, 0.00878906, 0.00683594,
  0.00341797, 0.00097656]])
    
    blurred_image = apply_gaussian_kernel(image, kernel)
    return blurred_image

def threshold_image(image, threshold):
    """
    Apply a threshold to the input image, converting pixel values above the threshold to 1 and below or equal to the threshold to 0.

    Parameters:
        image (numpy.ndarray): The input grayscale image.
        threshold (int): The threshold value.

    Returns:
        numpy.ndarray: The thresholded binary image.
    """
    copy = image.copy()
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            copy[i, j] = 1 if(copy[i, j] >= threshold) else 0
    return copy



def image_opening(binary_image, iterations=1):
    """
    Perform morphological opening operation on the binary image.

    Parameters:
        binary_image (numpy.ndarray): The input binary image.
        iterations (int, optional): Number of iterations for opening operation. Defaults to 1.

    Returns:
        numpy.ndarray: The binary image after morphological opening.
    """
    kernel = get_structuring_element(7)
    opened = binary_image
    for _ in range(iterations):
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)
    return opened

def image_closing(binary_image, iterations=1):
    """
    Perform morphological closing operation on the binary image.

    Parameters:
        binary_image (numpy.ndarray): The input binary image.
        iterations (int, optional): Number of iterations for closing operation. Defaults to 1.

    Returns:
        numpy.ndarray: The binary image after morphological closing.
    """
    kernel = get_structuring_element(7)
    closed = binary_image
    for _ in range(iterations):
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    return closed

def get_structuring_element(ksize):
    """
    Generate a structuring element (kernel) for morphological operations.

    Parameters:
        ksize (int): Size of the structuring element. Supported sizes are 3, 5, and 7.

    Returns:
        numpy.ndarray: The generated structuring element.
    
    Raises:
        ValueError: If an unsupported size is provided.
    """
    if ksize == 3:
        return np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)
    elif ksize == 5:
        return np.array([[0, 0, 1, 0, 0],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0]], dtype=np.uint8)
    elif ksize == 7:
        return np.array([[0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
    else:
        raise ValueError("Unsupported size")

def pad_image(image, pad_width):
    """
    Pad the input image with zeros.

    Parameters:
        image (numpy.ndarray): The input image.
        pad_width (tuple): Amount of padding along each axis.

    Returns:
        numpy.ndarray: The padded image.
    """
    return np.pad(image, pad_width, mode='constant', constant_values = 0)

def erode(image, kernel):
    """
    Perform erosion operation on the binary image using the specified kernel.

    Parameters:
        image (numpy.ndarray): The input binary image.
        kernel (numpy.ndarray): The structuring element for erosion.

    Returns:
        numpy.ndarray: The image after erosion.
    """
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = pad_image(image, ((pad_height, pad_height), (pad_width, pad_width)))
    output_image = np.zeros_like(image)
    
    for i in range(pad_height, pad_height + image.shape[0]):
        for j in range(pad_width, pad_width + image.shape[1]):
            region = padded_image[i - pad_height : i + pad_height + 1, j - pad_width:j + pad_width + 1]
            output_image[i - pad_height, j - pad_width] = np.min(region[kernel == 1])
    
    return output_image

def dilate(image, kernel):
    """
    Perform dilation operation on the binary image using the specified kernel.

    Parameters:
        image (numpy.ndarray): The input binary image.
        kernel (numpy.ndarray): The structuring element for dilation.

    Returns:
        numpy.ndarray: The image after dilation.
    """
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = pad_image(image, ((pad_height, pad_height), (pad_width, pad_width)))
    output_image = np.zeros_like(image)
    
    for i in range(pad_height, pad_height + image.shape[0]):
        for j in range(pad_width, pad_width + image.shape[1]):
            region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            output_image[i - pad_height, j - pad_width] = np.max(region[kernel == 1])
    
    return output_image

def image_opening_from_scratch(binary_image, ksize=3, iterations=1):
    """
    Perform morphological opening operation on the binary image using custom implementation.

    Parameters:
        binary_image (numpy.ndarray): The input binary image.
        ksize (int, optional): Size of the structuring element. Defaults to 3.
        iterations (int, optional): Number of iterations for opening operation. Defaults to 1.

    Returns:
        numpy.ndarray: The binary image after morphological opening.
    """
    kernel = get_structuring_element(ksize)
    opened_image = binary_image
    
    for _ in range(iterations):
        eroded_image = erode(opened_image, kernel)
        opened_image = dilate(eroded_image, kernel)
    
    return opened_image

def image_closing_from_scratch(binary_image, ksize=3, iterations=1):
    """
    Perform morphological closing operation on the binary image using custom implementation.

    Parameters:
        binary_image (numpy.ndarray): The input binary image.
        ksize (int, optional): Size of the structuring element. Defaults to 3.
        iterations (int, optional): Number of iterations for closing operation. Defaults to 1.

    Returns:
        numpy.ndarray: The binary image after morphological closing.
    """
    kernel = get_structuring_element(ksize)
    closed_image = binary_image
    
    for _ in range(iterations):
        dilated_image = dilate(closed_image, kernel)
        closed_image = erode(dilated_image, kernel)
    
    return closed_image

def lower_threshold(v, sigma=0.33):
    """
    Compute the lower threshold value for Canny edge detection based on the given intensity value and a sigma factor.

    Parameters:
        v (int): The intensity value.
        sigma (float, optional): The sigma factor to adjust the threshold. Defaults to 0.33.

    Returns:
        int: The lower threshold value.
    """
    return int(max(0, (1.0 - sigma) * v))

def upper_threshold(v, sigma=0.33):
    """
    Compute the upper threshold value for Canny edge detection based on the given intensity value and a sigma factor.

    Parameters:
        v (int): The intensity value.
        sigma (float, optional): The sigma factor to adjust the threshold. Defaults to 0.33.

    Returns:
        int: The upper threshold value.
    """
    return int(min(255, (1.0 + sigma) * v))

def median(image):
    """
    Compute the median value of pixel intensities in the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        float: The median pixel intensity value.
    """
    return np.median(image)

def auto_canny(image, sigma=0.33):
    """
    Perform automatic Canny edge detection on the input image.

    Parameters:
        image (numpy.ndarray): The input grayscale image.
        sigma (float, optional): The sigma factor for computing thresholds. Defaults to 0.33.

    Returns:
        numpy.ndarray: The binary edge image.
    """
    v = median(image)
    lower = lower_threshold(v, sigma)
    upper = upper_threshold(v, sigma)
    edged = cv2.Canny(image, lower, upper)
    return edged

def intersection_percentage(image, contour1, contour2):
    """
    Calculate the intersection percentage between two contours in the image.

    Parameters:
        image (numpy.ndarray): The input image.
        contour1 (numpy.ndarray): The first contour.
        contour2 (numpy.ndarray): The second contour.

    Returns:
        tuple: The intersection percentage of contour1 with contour2 and vice versa.
    """

    mask1 = np.zeros_like(image)
    mask2 = np.zeros_like(image)

    (x1, y1), radius1 = cv2.minEnclosingCircle(contour1)
    (x2, y2), radius2 = cv2.minEnclosingCircle(contour2)

    cv2.circle(mask1, (int(x1), int(y1)), int(radius1), (255), -1)
    cv2.circle(mask2, (int(x2), int(y2)), int(radius2), (255), -1)

    intersection = cv2.bitwise_and(mask1, mask2)

    intersection_area = np.count_nonzero(intersection)

    area1 = np.count_nonzero(mask1)
    area2 = np.count_nonzero(mask2)

    intersection_percentage1 = intersection_area / area1 if area1 != 0 else 0
    intersection_percentage2 = intersection_area / area2 if area2 != 0 else 0

    return intersection_percentage1, intersection_percentage2

def eliminate_intersecting_contours(image, contours):
    """
    Eliminate intersecting contours from the list of contours.

    Parameters:
        image (numpy.ndarray): The input image.
        contours (list): List of contours to be filtered.

    Returns:
        list: List of remaining non-intersecting contours.
    """
    remaining_contours = contours.copy()  # Initialize remaining contours with all contours

    i = 0
    while i < len(remaining_contours):
        contour1 = remaining_contours[i]
        should_keep = True

        j = 0
        while j < len(remaining_contours):
            contour2 = remaining_contours[j]

            if i != j:  # Ensure we're not comparing the contour with itself
                intersection_percentage1, intersection_percentage2 = intersection_percentage(image, contour1, contour2)

                # If either contour has more than 80% intersection with another contour, eliminate it
                if intersection_percentage1 >= 0.7 :
                    should_keep = False
                    break

            j += 1

        if not should_keep:
            del remaining_contours[i]  # Remove the contour from the list if it should not be kept
        else:
            i += 1

    return remaining_contours


def process_image(image):
    """
    Process an image to find contours.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        list: List of contours found in the image.
    """
    closed = image_closing(image, 3)
    edges = auto_canny(closed)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contours_canny(image):
    """
    Find contours in an image using Canny edge detection.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        list: List of contours found using Canny edge detection.
    """
    # gray = gray_scale(image)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    adap_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(adap_thresh, 50, 150)

    canny_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return canny_contours


