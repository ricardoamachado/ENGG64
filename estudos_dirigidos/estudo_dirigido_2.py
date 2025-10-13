import cv2 as cv
import seaborn as sns
import matplotlib.pyplot as plt
def main():
    #Configuração do estilo do plot.
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.dpi'] = 100
    sns.set_theme(style="whitegrid")
    # Carregando a imagem em escala de cinza.
    img = cv.imread('estudos_dirigidos/figuras/penguins.png', cv.IMREAD_GRAYSCALE)
    # Calculo e plot do histograma da imagem.
    histogram = cv.calcHist([img],[0],None,[256],[0,256])
    sns.lineplot(histogram)
    #Com base no histograma, escolhemos o threadhold como 100. 
    threshold_value = 100
    # Adicionando uma linha vertical no valor do threshold escolhido.
    plt.axvline(x=threshold_value, color='r', linestyle='--')
    # Configurando os rótulos e título do gráfico
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Número de Pixels')
    plt.title('Histograma da Imagem')
    plt.legend(['Histograma'])
    plt.tight_layout()
    plt.show()
    # Aplicando o thresholding na imagem.
    ret,thresh1 = cv.threshold(img,threshold_value,255,cv.THRESH_BINARY)
    cv.imshow('Imagem Original', img)
    cv.imshow('Imagem com Threshold', thresh1)
    cv.waitKey(0)

if __name__ == "__main__":
    main()
