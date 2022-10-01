import os
import glob
import cv2

local_path = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats'
files = glob.glob(os.path.join(local_path, 'train', '*.*.jpg'))
print(len(files))

c = 20
faceCascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
for file in files[:5]:
    img_bgr = cv2.imread(file, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detectedFaces = faceCascade.detectMultiScale(img_rgb, 1.1, 4)
    print(file, detectedFaces)
    # for (length,breadth,width,height) in detectedFaces:
    for (x, y, w, h) in detectedFaces:
        cv2.rectangle(img_rgb, (x - c - 1, y - c - 1), (x + w + c + 1, y + h + c + 1), (255, 0, 0), 1)
        (name, ext) = os.path.splitext(file)
        print(name, ext)

        # extract the file name and extension
        # file_name = split_tup[0]
        #file_extension = split_tup[1]
        #cv2.imwrite('roi.png', img_rgb[y - c:y + h + c, x - c:x + w + c])