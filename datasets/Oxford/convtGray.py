import cv2, os

list_file = []
for files in os.listdir('image_color'):
    if files.endswith('.png'):
        list_file += [files]

for files in list_file:
    img = cv2.imread('image_color/' + files)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('image_gray/' + files, img_gray)

