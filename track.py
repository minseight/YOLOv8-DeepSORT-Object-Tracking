import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort import DeepSort

try:
    from google.colab.patches import cv2_imshow
except ImportError:
    cv2_imshow = cv2.imshow

# 변수 초기화
unique_track_ids = set()
frames = []
i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

# DeepSORT 초기화
tracker = DeepSort()

# YOLO 모델 로드
model = YOLO("yolov8n.pt")  # 원하는 .pt 경로로 수정 가능

# 비디오 캡처
cap = cv2.VideoCapture("test3.mp4")  # 또는 0 (웹캠)

# 출력 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# 클래스 이름 정의
class_names = model.model.names  # YOLOv8이 학습한 클래스명

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패. 종료합니다.")
            break

        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = og_frame.copy()

        # YOLO 추론 실행
        results = model(frame_rgb, device=0, classes=0, conf=0.5)

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.cls is None:
                continue

            cls = boxes.cls.tolist()
            conf = boxes.conf.detach().cpu().numpy()
            xyxy = boxes.xyxy.detach().cpu().numpy()
            xywh = boxes.xywh.cpu().numpy()

            # DeepSORT 추적 업데이트
            tracks = tracker.update(xywh, conf, og_frame)

            for track in tracker.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = track.to_tlbr()
                w, h = int(x2 - x1), int(y2 - y1)

                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][track_id % 3]

                # 바운딩 박스 + ID 그리기
                cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                class_name = class_names[int(cls[0])] if cls else "Object"
                cv2.putText(og_frame, f"ID: {class_name}-{track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                unique_track_ids.add(track_id)

        # 사람 수 카운트 + FPS 표시
        person_count = len(unique_track_ids)
        current_time = time.perf_counter()
        elapsed = current_time - start_time
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        cv2.putText(og_frame, f"Count: {person_count}  FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 프레임 저장 + 출력
        frames.append(og_frame)
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
        cv2_imshow(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

