# ENGG64 – Visão Computacional
# Estudo Dirigido 3 - Detecção de Bordas
# Aluno: Ricardo Machado

import cv2 as cv
import numpy as np
def main():
    
    img = cv.imread('estudos_dirigidos/figuras/penguins.png', cv.IMREAD_GRAYSCALE)
    # Definindo o kernel de Sobel na vertical e horizontal.
    Kv = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
)
    Ku = np.transpose(Kv)
    # Aplicando os kernels na imagem pela convolução.
    Iv = cv.filter2D(img, -1, Kv)
    Iu = cv.filter2D(img, -1, Ku)
    # Calculando a imagem das bordas.
    I = np.hypot(Iu, Iv)  # noqa: E741
    I = np.uint8(I / I.max() * 255) # noqa: E741
    #Plot da imagem das bordas e das imagens dos gradientes.
    cv.imshow('Imagem das Bordas', I)
    cv.imshow('Gradiente Vertical', Iv)
    cv.imshow('Gradiente Horizontal', Iu)
    cv.waitKey(0)

if __name__ == "__main__":
    main()