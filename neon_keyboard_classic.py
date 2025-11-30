
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Классическая раскладка без SPACE, BACKSPACE, ENTER
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L",":"],
    ["Z", "X", "C", "V", "B", "N", "M",",",".","/"]
]

finalText = ""
pressed_key = None
press_time = 0

start_color = (245, 241, 100) # синий
end_color = (225, 101, 243) # Синий
time_counter = 0

def interpolate_color(color1, color2, factor):
    return (
    int(color1[0] + (color2[0] - color1[0]) * factor),
    int(color1[1] + (color2[1] - color1[1]) * factor),
    int(color1[2] + (color2[2] - color1[2]) * factor)
    )

def drawAll(frame, buttonList):
    global time_counter
    for i, button in enumerate(buttonList):
        x, y = button["pos"]
        w, h = button["size"]
        key = button["key"]

        wave = math.sin(time_counter + i * 0.3)
        factor = (wave + 1) / 2
        color = interpolate_color(start_color, end_color, factor)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, key, (x + 15, y + 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    return frame

# Создаем список кнопок
buttonList = []
start_x, start_y = 100, 100
for i, row in enumerate(keys):
    x_offset = 0
    for key in row:
        w = 85
        pos = (start_x + x_offset, start_y + i * 100)
        buttonList.append({"key": key, "pos": pos, "size": (w, 85)})
        x_offset += w + 10

def get_finger_tip(hand_landmarks, img_shape):
    h, w, _ = img_shape
    x = int(hand_landmarks.landmark[8].x * w)
    y = int(hand_landmarks.landmark[8].y * h)
    return x, y

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    time_counter += 0.05
    img = drawAll(img, buttonList)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        # Настраиваем стиль: чёрные точки и (опционально) светлые линии
        blue_dots = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
        light_lines = mp.solutions.drawing_utils.DrawingSpec(color=(245, 241, 100), thickness=1)

        mpDraw.draw_landmarks(
            img,
            handLms,
            mpHands.HAND_CONNECTIONS,
            landmark_drawing_spec=blue_dots,
            connection_drawing_spec=light_lines
        )
        x, y = get_finger_tip(handLms, img.shape)
        cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 8, (245, 241, 100), -1)

        for button in buttonList:
            bx, by = button["pos"]
            bw, bh = button["size"]
            key = button["key"]
            if bx < x < bx + bw and by < y < by + bh:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 5)
                if pressed_key != key:
                    pressed_key = key
                    press_time = time.time()
                else:
                    if time.time() - press_time > 0.7:
                        finalText += key
                        pyautogui.write(key.lower())
                        pressed_key = None
                        press_time = 0
                break
        else:
            pressed_key = None
            press_time = 0

    # Отображаем текст
    cv2.rectangle(img, (100, 500), (990, 580), (0, 0, 0), -1)
    cv2.putText(img, finalText, (110, 560), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.imshow("Neon Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
