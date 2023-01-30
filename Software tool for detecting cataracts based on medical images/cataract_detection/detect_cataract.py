import cv2
import sys

imagePath = sys.argv[1]
cascPath = "cascade.xml"

pedsCascade =  cv2.CascadeClassifier(cascPath)

# Читаємо зображення
image = cv2.imread(imagePath)
# resized_img = cv2.resize(image, (128, 128))
# Конвертація зображення у сірий колір
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Виялення кришталика

catarat = pedsCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=2,
        minSize=(50, 50)
)

print("Знайдено {0} катаракти!".format(len(catarat)))

# Створення рамки навколо кришталка
for (x, y, w, h) in catarat:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow("Faces found", image)
status = cv2.imwrite('catarat_saved.jpg', image)
print ("Опрацьоване медичне зображення збережене у файловій системі : ",status)
