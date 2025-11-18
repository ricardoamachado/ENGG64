# ENGG64 – Visão Computacional
# Estudo Dirigido 5 - Operações Diádicas
# Aluno: Ricardo Machado

import cv2
# Leitura do vídeo
cap = cv2.VideoCapture('estudos_dirigidos/videos/traffic_sequence.mpeg')
ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret:
    raise Exception("Não foi possível ler o vídeo.")

while ret:
    # Cálculo da diferença entre frames consecutivos.
    diff = cv2.absdiff(frame1, frame2)
    cv2.imshow('Difference', diff)

    frame1 = frame2
    ret, frame2 = cap.read()
    # Sai do loop se 'q' for pressionado.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()