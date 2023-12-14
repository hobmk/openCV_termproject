import cv2
import numpy as np

def make_transparent(image_path, target_color=(255, 255, 255)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 이미지가 3채널이면서 알파 채널이 없는 경우, 알파 채널 추가
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 지정된 색상과 일치하는 부분을 투명하게 만들기
    mask = np.all(img[:, :, :3] == target_color, axis=-1)
    img[mask, 3] = 0

    return img

# 얼굴 필터 이미지들 로드
face_filter_1 = make_transparent('test_sticker.png')
face_filter_2 = make_transparent('add_sticker.png')
face_filter_3 = make_transparent('add_sticker2.png')  # 새로운 스티커 추가

# 현재 사용 중인 얼굴 필터
current_face_filter = face_filter_1

# 얼굴 탐지기
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
        # 얼굴 영역 확장
        offset = 20  # 얼굴 영역을 좀 더 크게 설정
        face_roi = frame_flipped[max(0, y - offset):min(frame_flipped.shape[0], y + h + offset),
                                  max(0, x - offset):min(frame_flipped.shape[1], x + w + offset)]

        # 현재 얼굴 필터 이미지 크기 조절 (크기 조절을 통해 스티커 크기 조절)
        if current_face_filter is face_filter_1:
            sticker_resized = cv2.resize(current_face_filter, (w + 2 * offset, h + 2 * offset))
        else:
            # 새로운 스티커 크기 조절
            sticker_resized = cv2.resize(current_face_filter, (w + 2 * offset, h + 2 * offset))

        # 알파 채널을 이용한 얼굴 필터 합성
        alpha_s = sticker_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        # 얼굴 영역과 얼굴 필터 크기 일치시키기
        sticker_resized = cv2.resize(sticker_resized, (face_roi.shape[1], face_roi.shape[0]))

        # 알파 채널을 이용한 얼굴 필터 합성
        for c in range(0, 3):
            # 크기가 일치하도록 스티커와 얼굴 영역 크기 조절
            face_roi[:, :, c] = (alpha_s * sticker_resized[:, :, c] +
                                 alpha_l * face_roi[:, :, c])

    # 프레임 출력
    cv2.imshow('Video', frame_flipped)

    # 키 이벤트 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # 'q' 키를 누르면 루프를 빠져나와 프로그램 종료
        break
    elif key == ord('1'):
        # '1' 키를 누를 때마다 얼굴 필터 변경
        if current_face_filter is face_filter_1:
            current_face_filter = face_filter_2
        elif current_face_filter is face_filter_2:
            current_face_filter = face_filter_3
        else:
            current_face_filter = face_filter_1

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
