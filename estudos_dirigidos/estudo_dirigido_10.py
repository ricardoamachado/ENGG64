# ENGG64 – Visão Computacional
# Estudo Dirigido 10 - Transformada de Hough.
# Aluno: Ricardo Machado

import cv2 as cv
import numpy as np

def main():
    img = cv.imread('estudos_dirigidos/figuras/church.jpg',cv.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError("Imagem não encontrada.")
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Detecção de bordas usando Canny
    bordas = cv.Canny(img_grayscale, 50, 150, apertureSize=3)
    # Aplicar Transformada de Hough
    linhas = cv.HoughLines(bordas, 1, np.pi / 180, 200)
    linhas_prob = cv.HoughLinesP(bordas, 1, np.pi / 180, 70, None, 50, 10)
    img_prob = img.copy()
    # Desenhar linhas detectadas - Hough
    if linhas is not None:
        for rho, theta in linhas[:,0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv.imshow('Imagem com Linhas Detectadas - Hough', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # Desenhar linhas detectadas - Hough Probabilístico
    if linhas_prob is not None:
        for x1, y1, x2, y2 in linhas_prob[:,0]:
            cv.line(img_prob, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv.imshow('Imagem com Linhas Detectadas - Hough Probabilístico', img_prob)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()