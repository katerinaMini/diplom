import cv2 
import numpy as np
import math
from twilio.rest import Client 

# Учетные данные Twilio
#account_sid = 'your_account_sid'
#auth_token = 'your_auth_token'
#twilio_phone_number = 'your_twilio_phone_number'
#recipient_phone_number = 'your_recipient_phone_number'

# Функция для отправки SMS
#def send_sms(message):
    #client = Client(account_sid, auth_token)
    #client.messages.create(
     #   body=message,
      #  from_=twilio_phone_number,
       # to=recipient_phone_number
  #  )

# Загружаем модель для детектирования людей
net = cv2.dnn.readNetFromCaffe('model\deploy.prototxt', 'model\mobilenet_iter_73000.caffemodel')

# Параметры камеры или видеопотока
rtsp_url = "rtsp://username:password@192.168.1.100:554/stream"
cap = cv2.VideoCapture(0)

# Задаем параметры для записи видео
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для сохранения видео
output_filename = 'output_video.avi'  # Имя выходного файла
fps = 20.0  # Частота кадров (подстраивается под реальную камеру)
frame_size = (int(cap.get(3)), int(cap.get(4)))  # Размер кадра (ширина, высота)
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

# Задаем зону интереса для обнаружения людей
roi_top_left = (100, 100)
roi_bottom_right = (500, 500)

# Координаты контрольных точек для ворот
point1 = (600, 300)
point2 = (700, 300)

# Начальное расстояние между воротами
initial_distance = math.dist(point1, point2)

# Флаг для отслеживания отправки уведомлений
notification_sent_for_gate = False
notification_sent_for_person = False

# Основной цикл для чтения и обработки кадров
while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # Преобразуем кадр для подачи в нейронную сеть
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Отрисовываем зону интереса (ROI)
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

    # Проходим по всем обнаруженным объектам (людям)
    person_detected = False
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Фильтруем только уверенные распознавания
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Проверка, попадает ли человек в зону интереса (ROI)
            person_in_roi = (startX > roi_top_left[0] and startY > roi_top_left[1] and endX < roi_bottom_right[0] and endY < roi_bottom_right[1])

            if person_in_roi:
                label = f"Person: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                person_detected = True

                # Если человек в зоне интереса и уведомление еще не отправлено
                #if not notification_sent_for_person:
                 #   send_sms("Внимание: обнаружен человек в зоне интереса!")
                  #  print("SMS отправлено: обнаружен человек в зоне!")
                   # notification_sent_for_person = True

    if not person_detected:
        notification_sent_for_person = False

    # Определяем положение ворот и проверяем, открыты ли они
    cv2.circle(frame, point1, 5, (255, 0, 0), -1)
    cv2.circle(frame, point2, 5, (255, 0, 0), -1)

    current_distance = math.dist(point1, point2)

    if current_distance > initial_distance + 20:  # 20 - допустимый порог
        cv2.putText(frame, "Ворота открыты!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
       # # Если ворота открыты и уведомление еще не отправлено
      #  if not notification_sent_for_gate:
       #     send_sms("Внимание: ворота открыты!")
        #    print("SMS отправлено: ворота открыты!")
         #   notification_sent_for_gate = True
   # else:
    #    cv2.putText(frame, "Ворота закрыты", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
     #   notification_sent_for_gate = False

    # Сохраняем обработанный кадр в видеофайл
    out.write(frame)

    # Отображаем кадр
    cv2.imshow("Frame", frame)

    # Завершаем программу по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()  # Закрываем файл с видео
cv2.destroyAllWindows()
