# ENGG64 – Visão Computacional
# Estudo Dirigido 8 - Classificação por Cores.
# Aluno: Ricardo Machado

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Configurações de plotagem
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (7, 5)
    plt.rcParams['figure.dpi'] = 150
    img = cv2.imread('estudos_dirigidos/figuras/yellowtargets.png',cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError("Imagem não encontrada.")
    # Converter imagem para float
    img_float = img.astype(np.float32) / 255.0
    # Separar canais
    R, G, B = cv2.split(img_float)
    # Calcular componentes cromáticas
    r = np.ma.divide(R, R + G + B)
    g = np.ma.divide(G, R + G + B)
    b = np.ma.divide(B, R + G + B)
    # Imagem cromática
    chromatic_img = cv2.merge((r, g, b))
    cv2.imshow('Imagem no plano de cromaticidade', chromatic_img)
    # Aplicando filtros de suavização
    chromatic_img = cv2.blur(chromatic_img, (5, 5))
    chromatic_img = cv2.GaussianBlur(chromatic_img, (5, 5), 2)
    cv2.imshow('Imagem suavizada', chromatic_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Uso de k-means para segmentação
    K = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(chromatic_img,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Converter centro para uint8
    center = np.uint8(center * 255)
    # Mapear rótulos para centroide
    segmented_data = center[label.flatten()]
    segmented_img = segmented_data.reshape((img.shape))
    cv2.imshow('Imagem Segmentada', segmented_img)
    cv2.imwrite('estudos_dirigidos/figuras/yellowtargets_segmented.png', segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Visualizar canais da imagem no plano da cromaticidade. Os alvos amarelos são muito escuros em no canal azul.
    b_channel, g_channel, r_channel = cv2.split(chromatic_img)
    cv2.imshow('Canal r da Imagem Segmentada', r_channel)
    cv2.imshow('Canal g da Imagem Segmentada', g_channel)
    cv2.imshow('Canal b da Imagem Segmentada', b_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(chromatic_img)
    histogram = cv2.calcHist([b_channel],[0],None,[256],[0,1])
    sns.lineplot(histogram)
    plt.title('Histograma do Canal b da Imagem Segmentada')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Número de Pixels')
    plt.tight_layout()
    plt.show()
    # Com base no histograma, escolhemos o threadhold como 58/255, pois a imagem está em formato float.
    threshold_value = 58/255
    # Aplicando o thresholding no canal b da imagem segmentada.
    ret,thresh1 = cv2.threshold(b_channel,threshold_value,1,cv2.THRESH_BINARY)
    cv2.imshow('Imagem binária', thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Plotagem do plano de cromaticidade original e segmentado na mesma figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Preparar dados (preencher máscaras com 0)
    r_vals = np.ma.filled(r, 0).flatten()
    g_vals = np.ma.filled(g, 0).flatten()
    b_vals = np.ma.filled(b, 0).flatten()

    # Primeiro subplot: plano de cromaticidade original
    sc0 = axes[0].scatter(r_vals, g_vals, c=b_vals, s=1, cmap='viridis')
    axes[0].set_title('Plano de Cromaticidade (r vs g) - Original')
    axes[0].set_xlabel('Componente r')
    axes[0].set_ylabel('Componente g')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    fig.colorbar(sc0, ax=axes[0], label='Componente b')

    # Segundo subplot: plano de cromaticidade segmentado
    r_seg = segmented_img[:, :, 0].flatten() / 255.0
    g_seg = segmented_img[:, :, 1].flatten() / 255.0
    b_seg = segmented_img[:, :, 2].flatten() / 255.0
    sc1 = axes[1].scatter(r_seg, g_seg, c=b_seg, s=1, cmap='viridis')
    axes[1].set_title('Plano de Cromaticidade (r vs g) - Segmentado')
    axes[1].set_xlabel('Componente r')
    axes[1].set_ylabel('Componente g')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    fig.colorbar(sc1, ax=axes[1], label='Componente b')

    plt.tight_layout()
    plt.show()
    #TODO: Determinar os centroides dos alvos amarelos no plano de cromaticidade.
if __name__ == "__main__":
    main()