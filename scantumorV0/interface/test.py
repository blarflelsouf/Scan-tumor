
import utils
import cv2



path_data = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/Training'

a = utils.load_data_dataframe(path_data)

print(a['images_paths'][0])
im = cv2.imread(a['images_paths'][0])
print(im.shape[:2])
