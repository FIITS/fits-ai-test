import cv2
import numpy as np
from sklearn.cluster import KMeans

# Haar Cascade 분류기를 사용하여 얼굴을 인식하기 위한 분류기 로딩
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠을 사용하기 위해 VideoCapture 객체 생성
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 나타냄. 다른 카메라를 사용하려면 숫자를 조절하거나 경로를 입력하세요.

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 그레이스케일(흑백)로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 검출된 얼굴에 사각형 표시 및 personal_color 출력
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_roi = frame[y:y + h, x:x + w]

        # 이미지 면적의 8%만큼 중앙을 또 자르기
        center_x, center_y = x + w // 2, y + h // 2
        crop_percentage = 0.08
        crop_size = int(min(w, h) * crop_percentage)

        cropped_face = frame[center_y - crop_size // 2:center_y + crop_size // 2,
                            center_x - crop_size // 2:center_x + crop_size // 2]

        # 대표색 추출을 위해 이미지를 1차원 배열로 변환
        pixels = cropped_face.reshape((-1, 3))

        # KMeans를 사용하여 대표색 클러스터링
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(pixels)

        # 대표색 얻기
        representative_color = kmeans.cluster_centers_.astype(int)[0]
        representative_color_hsv = cv2.cvtColor(np.uint8([[representative_color]]), cv2.COLOR_BGR2HSV)[0][0]

        # 대표색을 HSV 형식으로 출력
        print("대표색 (HSV):", representative_color_hsv)
        
        # 대표색의 명도, 채도 퍼센트로 변환
        hue, saturation, value = representative_color_hsv
        hue_p_v = (value / 255) * 100
        saturation_p_v = (saturation / 255) * 100
        
        # 대표색의 명도, 채도 출력
        print("명도 (Value):", hue_p_v, "%")
        print("채도 (Saturation):", saturation_p_v, "%")

        # 개인 색상 판별
        personal_color = "none"
        if hue_p_v >= 50:
            if saturation_p_v >= 25:
                personal_color = "Spring, warm tone" # 봄, 웜톤
            else:
                personal_color = "Summer. Cool tone" # 여름, 쿨톤
        if hue_p_v < 50:
            if saturation_p_v < 25:
                personal_color = "Autumn, warm tone" # 가을, 웜톤
            else:
                personal_color = "Winter. Cool tone" # 겨울, 쿨톤

        # 퍼스널 컬러 출력
        print("퍼스널 컬러 (Personal color):", personal_color)

        # 얼굴에 대한 사각형 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # personal_color 값을 화면에 표시 (UTF-8 인코딩)
        personal_color_text = f'Personal Color: {personal_color}'
        cv2.putText(frame, personal_color_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 결과 출력
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업이 끝난 후 웹캠 해제
cap.release()
cv2.destroyAllWindows()