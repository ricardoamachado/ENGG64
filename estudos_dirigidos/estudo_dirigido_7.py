# ENGG64 – Visão Computacional
# Estudo Dirigido 7 – Histograma Imagens Coloridas
# Aluno: Ricardo Machado

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    image = cv2.imread('estudos_dirigidos/figuras/flowers9.png')
    # Separando os canais BGR.
    b_channel, g_channel, r_channel = cv2.split(image)
    # Calculando os histogramas para cada canal.
    hist_b = cv2.calcHist([b_channel],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([g_channel],[0],None,[256],[0,256])
    hist_r = cv2.calcHist([r_channel],[0],None,[256],[0,256])
    #Configuração do estilo do plot.
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    sns.set_theme(style="whitegrid")
    # Plotando os histogramas.
    plt.figure(1)
    plt.subplot(3,1,1)
    sns.lineplot(hist_b, color='blue')
    plt.title('Histograma do Canal Azul')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Número de Pixels')
    plt.subplot(3,1,2)
    sns.lineplot(hist_g, color='green')
    plt.title('Histograma do Canal Verde')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Número de Pixels')
    plt.subplot(3,1,3)
    sns.lineplot(hist_r, color='red')
    plt.title('Histograma do Canal Vermelho')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Número de Pixels')
    plt.tight_layout()
    plt.show()
    # Visualizando cada canal separadamente.
    cv2.imshow('Canal Azul', b_channel)
    cv2.imshow('Canal Verde', g_channel)
    cv2.imshow('Canal Vermelho', r_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()