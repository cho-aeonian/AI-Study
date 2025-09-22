from PIL import Image

img = Image.open(r"C:\Users\U2SR11\Desktop\mediapipe+face\nose.png")
print(img.mode)  # 출력이 'RGBA' 여야 함
