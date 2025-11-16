# ENGG64 – Visão Computacional
# Estudo Dirigido 6 - Kernel
# Aluno: Ricardo Machado

import cv2 as cv
import numpy as np

def main():
    img = cv.imread('estudos_dirigidos/figuras/flowers4.png', cv.IMREAD_GRAYSCALE)
    # Convertendo a imagem para double e normalizando os valores entre 0 e 1.
    img_float = img.astype(np.float64) / 255.0
    # Definindo o kernel média 15x15.
    kU = np.ones((15, 15)) / 15**2
    imU = cv.filter2D(img_float, -1, kU)
    # Definindo o kernel gaussiano sigma = 5 e meia largura = 8.
    MEIA_LARGURA = 8
    kernel_size = (2 * MEIA_LARGURA) + 1
    kG = cv.getGaussianKernel(kernel_size, 5)
    imG = cv.filter2D(img_float, -1, kG)
    # Visualização das imagens.
    cv.imshow('Imagem Float', img_float)
    cv.imshow('Imagem Media', imU)
    cv.imshow('Imagem Gaussiana', imG)
    cv.waitKey(0)
if __name__ == "__main__":
    main()