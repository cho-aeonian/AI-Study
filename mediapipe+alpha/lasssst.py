import cv2
import numpy as np
import mediapipe as mp

#흰색 배경 투명하게 바꾸기
def make_white_background_transparent(img, threshold=240):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(img)
    white_mask = (b >= threshold) & (g >= threshold) & (r >= threshold)
    a[white_mask] = 0
    img = cv2.merge((b, g, r, a))
    return img

left_ear_path = r"C:\Users\U2SR11\Desktop\mediapipe+face\left_ear.png"
right_ear_path = r"C:\Users\U2SR11\Desktop\mediapipe+face\right_ear.png"
nose_path = r"C:\Users\U2SR11\Desktop\mediapipe+face\nose.png"

left_ear = cv2.imread(left_ear_path, cv2.IMREAD_UNCHANGED)
right_ear = cv2.imread(right_ear_path, cv2.IMREAD_UNCHANGED)
nose = cv2.imread(nose_path, cv2.IMREAD_UNCHANGED)

#투명 배경 전처리
if left_ear.shape[2] != 4 or np.all(left_ear[:, :, 3] == 255):
    left_ear = make_white_background_transparent(left_ear)
if right_ear.shape[2] != 4 or np.all(right_ear[:, :, 3] == 255):
    right_ear = make_white_background_transparent(right_ear)
if nose.shape[2] != 4 or np.all(nose[:, :, 3] == 255):
    nose = make_white_background_transparent(nose)

#알파값 추출해서 투명 배경 합성
def overlay_transparent(background, overlay, x, y, scale=1.0):
    w = int(overlay.shape[1] * scale)
    h = int(overlay.shape[0] * scale)
    if w == 0 or h == 0:
        return background
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    b, g, r, a = cv2.split(overlay_resized)
    overlay_color = cv2.merge((b, g, r))
    mask = a / 255.0

    h_bg, w_bg = background.shape[:2]

    x = max(0, x)
    y = max(0, y)

    if x + w > w_bg:
        w = w_bg - x
        overlay_color = overlay_color[:, :w]
        mask = mask[:, :w]
    if y + h > h_bg:
        h = h_bg - y
        overlay_color = overlay_color[:h, :]
        mask = mask[:h, :]

    roi = background[y:y+h, x:x+w]

    for c in range(3):
        roi[:, :, c] = (mask * overlay_color[:, :, c] + (1 - mask) * roi[:, :, c])

    background[y:y+h, x:x+w] = roi
    return background

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

sticker_on = True #스티커 여부

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks and sticker_on:
        landmarks = results.multi_face_landmarks[0]

        #얼굴 특징점 추출
        left_ear_landmark = landmarks.landmark[234]  #왼쪽 귀
        right_ear_landmark = landmarks.landmark[454] #오른쪽 귀
        nose_landmark = landmarks.landmark[1]        #코 끝

        lx, ly = int(left_ear_landmark.x * w), int(left_ear_landmark.y * h)
        rx, ry = int(right_ear_landmark.x * w), int(right_ear_landmark.y * h)
        nx, ny = int(nose_landmark.x * w), int(nose_landmark.y * h)

        face_width = abs(rx - lx)

        ear_scale = face_width / left_ear.shape[1] * 1.1
        nose_scale = face_width / nose.shape[1] * 0.7

        #왼쪽 귀 조정 
        offset_x_ear_left =35
        offset_y_ear_left = -30

        #오른쪽 귀 조정
        offset_x_ear_right = 100
        offset_y_ear_right = -30

        #왼쪽 귀
        frame = overlay_transparent(frame, left_ear, 
                                lx - int(left_ear.shape[1] * ear_scale) + offset_x_ear_left,
                                ly - int(left_ear.shape[0] * ear_scale) + offset_y_ear_left, 
                                ear_scale)
        #오른쪽 귀
        frame = overlay_transparent(frame, right_ear, 
                                rx - int(right_ear.shape[1] * ear_scale) + offset_x_ear_right,
                                ry - int(right_ear.shape[0] * ear_scale) + offset_y_ear_right, 
                                ear_scale)
        #코
        frame = overlay_transparent(frame, nose, 
                                nx - int(nose.shape[1] * nose_scale/2),
                                ny - int(nose.shape[0] * nose_scale/2), 
                                nose_scale)

    cv2.imshow("Cat Face Sticker", frame)

    key = cv2.waitKey(1) & 0xFF
    #q 누르면 캠 화면+코드 전체 종료
    if key == ord('q'):
        break
    #s 누르면 스티커 여부 변수따라 on/off
    elif key == ord('s'):
        sticker_on = not sticker_on

cap.release()
cv2.destroyAllWindows()
