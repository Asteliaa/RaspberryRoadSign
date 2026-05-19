import cv2

frame = cv2.imread("images/image5.jpg")
points = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x={x}, y={y}")
        points.append((x, y))

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", on_click)

while True:
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Все точки:", points)
