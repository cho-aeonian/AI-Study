import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

#딥러닝 얼굴 검출 모델
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

tracker = None #얼굴 추적하는 트래커
initBB = None #맨 처음 5초 추적하는 바운딩박스
lost_frames = 0 #트래킹 리셋

start_time = time.time()
face_box = None #초기 얼굴값
max_confidence = 0 #초기 얼굴이랑 유사도 제일 높은 값

while True:
    ret, frame = cap.read() #프레임 읽어오기
    if not ret:
        break

    (h, w) = frame.shape[:2]
    current_time = time.time()
    elapsed = current_time - start_time

    if initBB is None:
        if elapsed < 5:
            # 딥러닝 모델 전처리
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                 #신뢰도 비교
                if confidence > 0.8 and confidence > max_confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    width = endX - startX
                    height = endY - startY
                    
                    if width < 30 or height < 30:
                        continue
                    
                    face_box = (startX, startY, width, height) #신뢰도 높은 얼굴 저장
                    max_confidence = confidence

            if face_box is not None:
                cv2.rectangle(frame, (face_box[0], face_box[1]), 
                              (face_box[0] + face_box[2], face_box[1] + face_box[3]), 
                              (255, 255, 0), 2)
                #남은 시간 표시
                cv2.putText(frame, f"Selecting target face... {int(5 - elapsed)}s left", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        else:
            if face_box is not None:
                tracker = cv2.TrackerKCF_create()
                initBB = face_box 
                tracker.init(frame, initBB)
                lost_frames = 0
                cv2.putText(frame, "Tracking initialized", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        success, box = tracker.update(frame)
        
        if success:
            lost_frames = 0
            (x, y, w_, h_) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking target face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            lost_frames += 1
            cv2.putText(frame, "Tracking lost...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            #일정 시간 이상 등록되었던 얼굴이 안 잡히면 값 초기화
            if lost_frames >= 100:
                initBB = None
                tracker = None
                lost_frames = 0
                start_time = time.time()
                face_box = None
                max_confidence = 0

    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
