# ENGG64 – Visão Computacional
# Estudo Dirigido 1 - Animação do Olho Lena.
# Aluno: Ricardo Machado

import cv2
import numpy as np
def blink_lena_eye(image_path: str, num_blinks: int = 3, open_duration_ms: int = 500, closed_duration_ms: int = 300):
    """
    Faz o olho da Lena piscar um número especificado de vezes.

    Args:
        image_path: O caminho para a imagem Lena
        num_blinks: O número de vezes que o olho deve piscar.
        open_duration_ms: Duração da exibição do olho aberto (em ms).
        closed_duration_ms: Duração da exibição do olho fechado (em ms).
    """
    img_original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img_original is None:
        raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")
    # Parametros para desenhar a elipse que simula o olho fechado. Definidos por tentativa e erro.
    center = (275, 260)
    axes = (22, 16)
    angle = 0
    start_angle = 0
    end_angle = 360
    # Cor da elipse baseada na média da região do olho.
    color = int(np.mean(img_original[250:285, 240:290])) 
    thickness = -1 
    img_closed = img_original.copy()
    
    # Simulação do olho fechado desenhando uma elipse sobre o olho.
    cv2.ellipse(img_closed, center, axes, angle, start_angle, end_angle, color, thickness)
    # Configuração da janela de exibição
    window_name = 'Lena'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"Iniciando {num_blinks} piscadas. Pressione 'q' para sair.")

    for i in range(num_blinks):
        # Definição do loop de exibição das imagens.
        cv2.imshow(window_name, img_original)
        if cv2.waitKey(open_duration_ms) & 0xFF == ord('q'):
            break
        cv2.imshow(window_name, img_closed)
        if cv2.waitKey(closed_duration_ms) & 0xFF == ord('q'):
            break
    # Exibe a imagem original após as piscadas.
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.imshow(window_name, img_original)
        print("Piscadas concluídas. Pressione qualquer tecla para fechar.")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def main():
    image_path = 'estudos_dirigidos/figuras/lena.pgm'
    blink_lena_eye(image_path, num_blinks=3, open_duration_ms=500, closed_duration_ms=300)

if __name__ == "__main__":
    main()