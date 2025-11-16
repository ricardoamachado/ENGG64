# ENGG64 – Visão Computacional
# Estudo Dirigido 4 - Processamento Monádico
# Aluno: Ricardo Machado

import cv2 as cv
import numpy as np

def main():
    img = cv.imread('estudos_dirigidos/figuras/lena.pgm', cv.IMREAD_GRAYSCALE)
    UINT8_MAX = 255
    # Convertendo a imagem para float32 e normalizando os valores entre 0 e 1.
    img_float = np.float32(img) / UINT8_MAX
    # Processamento de brilho.
    img_bright = np.clip(img_float + 0.25, 0, 1)
    # Processamento de contraste.
    img_contrast = np.clip((img_float * 2), 0, 1)
    # Imagem negativa
    img_negative = 1 - img_float
    #Posterização com 8 níveis.
    img_posterized = np.floor(img_float * 8) / 8
    # Verificando os valores mínimo e máximo da imagem do tipo float.
    print(f"Max: {img_float.max()}, Min: {img_float.min()}")
    # Visualizando as imagens.
    cv.imshow('Imagem Float', img_float)
    cv.imshow('Imagem com Brilho', img_bright)
    cv.imshow('Imagem com Contraste', img_contrast)
    cv.imshow('Imagem Negativa', img_negative)
    cv.imshow('Imagem Posterizada', img_posterized)
    cv.waitKey(0)
if __name__ == "__main__":
    main()