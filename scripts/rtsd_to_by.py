import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

ID_TO_BY = {
    0: "2.1",
    1: "1.23",
    2: "1.17",
    3: "3.24.1",
    4: "7.2.1",
    5: "5.20",
    6: "5.16.1",
    7: "5.12.1",
    8: "3.25.1",
    9: "5.33",
    10: "7.15",
    11: "2.2",
    12: "2.4",
    13: "7.13",
    14: "4.2.1",
    15: "1.20.3",
    16: "1.25",
    17: "3.4",
    18: "7.3.2",
    19: "3.4.1",
    20: "4.1.6",
    21: "4.2.3",
    22: "4.1.1",
    23: "1.33",
    24: "5.8.5",
    25: "3.27",
    26: "1.15",
    27: "4.1.2",
    28: "6.3",
    29: "7.1.1",
    30: "6.7",
    31: "5.8.3",
    32: "6.11",
    33: "1.19",
    34: "5.15",
    35: "7.1.4",
    36: "7.8",
    37: "1.16.1",
    38: "1.11.2",
    39: "6.6",
    40: "5.8.1",
    41: "6.2",
    42: "5.8.1",
    43: "6.12",
    44: "3.18.1",
    45: "5.6",
    46: "5.5",
    47: "6.4",
    48: "4.1.2",
    49: "7.2.2",
    50: "6.11",
    51: "1.22",
    52: "1.27",
    53: "2.3.2",
    54: "5.8.2",
    55: "1.8",
    56: "3.13",
    57: "2.3.1",
    58: "7.3.3",
    59: "2.3.3",
    60: "6.8",
    61: "1.11.1",
    62: "7.13",
    63: "1.12.2",
    64: "1.20.1",
    65: "1.12.1",
    66: "3.32",
    67: "2.5",
    68: "3.1",
    69: "4.8.2",
    70: "3.20",
    71: "3.2",
    72: "2.3.6",
    73: "5.22",
    74: "5.18",
    75: "2.3.5",
    76: "6.5",
    77: "7.4.1",
    78: "3.14",
    79: "1.2",
    80: "1.20.2",
    81: "4.1.4",
    82: "6.6",
    83: "7.1.3",
    84: "7.3.1",
    85: "4.3",
    86: "4.1.5",
    87: "7.2.3",
    88: "7.2.4",
    89: "1.31.1",
    90: "3.10",
    91: "4.2.2",
    92: "6.1",
    93: "3.28",
    94: "4.1.3",
    95: "5.4",
    96: "5.3",
    97: "5.8.2",
    98: "3.31",
    99: "6.2",
    100: "1.21",
    101: "3.21",
    102: "1.13",
    103: "1.14",
    104: "2.3.4",
    105: "4.8.3",
    106: "5.15",
    107: "2.6",
    108: "3.18.2",
    109: "4.1.3",
    110: "1.7",
    111: "3.19",
    112: "1.18",
    113: "2.7",
    114: "7.5.4",
    115: "5.8.7",
    116: "5.14",
    117: "5.21",
    118: "1.1",
    119: "5.15",
    120: "7.6.4",
    121: "7.15",
    122: "4.5.1",
    123: "3.11.1",
    124: "7.18",
    125: "7.4.4",
    126: "3.30",
    127: "5.7.1",
    128: "5.7.2",
    129: "1.5",
    130: "3.29",
    131: "6.11",
    132: "5.12.2",
    133: "3.16",
    134: "1.30",
    135: "5.9.1",
    136: "1.6",
    137: "7.6.2",
    138: "6.9",
    139: "3.12",
    140: "3.33",
    141: "7.4.3",
    142: "5.8.1",
    143: "7.14",
    144: "7.17",
    145: "3.6",
    146: "1.26",
    147: "7.5.2",
    148: "6.8",
    149: "5.17.1",
    150: "1.10",
    151: "7.16",
    152: "6.15",
    153: "7.14",
    154: "7.23"
}

class BySignDetector:

    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        logger.info(f"Загрузка модели: {model_path}")
        self.model = YOLO(model_path)

        self.colors = {}
        for cls_id in ID_TO_BY.keys():
            import random
            random.seed(cls_id)
            self.colors[cls_id] = (random.randint(
                0, 255), random.randint(0, 255), random.randint(0, 255))

    def detect(self, frame):
        return self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]

    def draw_predictions(self, frame, results):
        frame_copy = frame.copy()

        if results.boxes is None:
            return frame_copy

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            sign_info = ID_TO_BY.get(class_id)

            if sign_info:
                label = f"{sign_info['code']} {sign_info['desc']}"
                color = self.colors.get(class_id, (0, 255, 0))

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame_copy, (x1, y1 - 20),
                              (x1 + w, y1), color, -1)

                cv2.putText(frame_copy, f"{label} ({conf:.2f})",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
            else:
                logger.warning(f"Class ID {class_id} detected but not mapped!")

        return frame_copy
