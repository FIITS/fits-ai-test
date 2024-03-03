import cv2
import numpy as np

def personal_color_test(image_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"이미지를 읽을 수 없습니다. ({image_path})")

    # BGR을 HSV로 변환
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 피부색을 나타내는 HSV 범위 설정
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # 피부색 마스크 생성
    skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)

    # 피부색 영역만 남기기
    skin_color = cv2.bitwise_and(img, img, mask=skin_mask)

    # 피부색 대표색 계산
    avg_color = np.mean(skin_color, axis=(0, 1))

    # 대표색을 BGR 형식에서 HSV 형식으로 변환
    avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # HSV 형식으로 출력
    print("대표색 (HSV):", avg_color_hsv)

    # 대표색의 명도, 채도 퍼센트로 변환
    hue, saturation, value = avg_color_hsv
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

    return personal_color

# 이미지 파일 경로 리스트 생성
image_paths = ["1.png", "2.png", "3.png", "4.png", "5.png", "iu.png", "f.png", "me.png", "me2.png", "now.png"]

# 이미지들에 대해 피부색 추출 및 대표색 계산 수행
for path in image_paths:
    print(f"\nProcessing image: ./image/{path}")
    print(personal_color_test(f"./image/{path}"))
