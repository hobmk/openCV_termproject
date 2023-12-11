import cv2
import numpy as np


def make_transparent(image_path, output_path, target_color=(255, 255, 255)):

    # 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 지정된 색상과 일치하는 부분을 투명하게 만들기
    mask = np.all(img[:, :, :3] == target_color, axis=-1)

    # 알파 채널을 0으로 설정하여 투명하게 만들기
    img[mask, 3] = 0

    # 이미지 저장
    cv2.imwrite(output_path, img)


make_transparent('test_sticker.png', 'test_sticker_invi.png')

# 얼굴 탐지기
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 스티커 이미지 로드 (배경이 투명한 PNG 파일 사용)
sticker = cv2.imread('test_sticker_invi.png', cv2.IMREAD_UNCHANGED)

# 웹캠 캡쳐 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # 좌우 반전
    frame_flipped = cv2.flip(frame, 1)

    # 얼굴 탐지
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY), 1.3, 5)

    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_roi = frame_flipped[y:y+h, x:x+w]

        # 스티커 이미지 크기 조절
        sticker_resized = cv2.resize(sticker, (w, h))

        # 알파 채널을 이용한 스티커 합성
        alpha_s = sticker_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        # 얼굴 영역과 스티커 크기 일치시키기
        sticker_area = frame_flipped[y:y+h, x:x+w, :3]
        sticker_resized = cv2.resize(sticker_resized, (sticker_area.shape[1], sticker_area.shape[0]))

        # 알파 채널을 이용한 스티커 합성
        for c in range(0, 3):
            face_roi[:, :, c] = (alpha_s * sticker_resized[:, :, c] +
                                 alpha_l * face_roi[:, :, c])

    # 프레임 출력
    cv2.imshow('Video', frame_flipped)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
