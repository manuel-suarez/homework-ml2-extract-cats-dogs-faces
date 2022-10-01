import os
import glob
import cv2

local_path = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats'
files = glob.glob(os.path.join(local_path, 'train', '*.*.jpg'))
print(len(files))
