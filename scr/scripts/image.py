import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def draw_ru_box(img_bgr, label, conf, box, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_PATH, 20)

    text = f"{label} {conf:.2f}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
    draw.text((x1 + 4, y1 - th - 4), text, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

FAKE_DETECTIONS = [
    {
        "label": "4.6.1: Пешеходная дорожка",
        "conf":  0.92,
        "box":   (1157, 428, 1274, 561),   # большой правый
        "color": (0, 200, 0),
    },
    {
        "label": "5.18.1: Рекомендуемая скорость",
        "conf":  0.86,
        "box":   (851, 548, 915, 620),     # средний
        "color": (0, 150, 255),
    },
    {
        "label": "1.25: Дикие животные",
        "conf":  0.83,
        "box":   (779, 569, 839, 628),     # маленький левый
        "color": (200, 0, 180),
    },
]

frame = cv2.imread("images/image5.jpg")

for det in FAKE_DETECTIONS:
    frame = draw_ru_box(frame, det["label"], det["conf"],
                        det["box"], det["color"])

cv2.imwrite("result5.jpg", frame)
print("Готово → result5.jpg")
print(frame.shape)





