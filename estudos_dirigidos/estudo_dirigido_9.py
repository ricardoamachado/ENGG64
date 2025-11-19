# ENGG64 – Visão Computacional
# Estudo Dirigido 9 - Momentos.
# Aluno: Ricardo Machado

import cv2
import numpy as np

def main():
    img = cv2.imread('estudos_dirigidos/figuras/aviao_ed.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Imagem não encontrada.")
    # Determinando os momentos de ordem 0, 1 e 2.
    moments = cv2.moments(img)
    # Determinando a área.
    area = moments['m00']
    print(f"Área: {area}")
    # Determinando o centroide.
    x_centroide = moments['m10'] / moments['m00']
    y_centroide = moments['m01'] / moments['m00']
    centroide = (int(x_centroide), int(y_centroide))
    print(f"Centroide: {centroide}")
    # Construindo a matriz inercial.
    matriz_inercial = np.array([[moments['mu20'], moments['mu11']],
                                [moments['mu11'], moments['mu02']]])
    print(f"Matriz Inercial:\n{matriz_inercial}")
    # Calculando os autovalores e autovetores da matriz inercial.
    autovalores, autovetores = np.linalg.eig(matriz_inercial)
    # Determinação dos comprimentos dos eixos principais da elipse.
    a = 2 * np.sqrt(autovalores[0] / moments['m00'])
    b = 2 * np.sqrt(autovalores[1] / moments['m00'])
    print("Comprimentos dos Eixos da Elipse Equivalente:")
    print(f"Eixo maior (a): {a}, Eixo menor (b): {b}")
    # Calculando o ângulo de orientação da elipse.
    theta = np.arctan(autovetores[1,0] / autovetores[0,0])
    print(f"Ângulo de orientação (radianos): {theta}")
    theta = np.degrees(theta)
    print(f"Ângulo de orientação (graus): {theta}")
    # Desenhando a elipse sobre a imagem original.
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_color, centroide, 5, (255,255,0), -1)
    cv2.ellipse(img_color, centroide, (int(a), int(b)), theta, 0, 360, (255,255,0), 2)
    cv2.imshow('Imagem Original', img)
    cv2.imshow('Imagem com Elipse Equivalente', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()