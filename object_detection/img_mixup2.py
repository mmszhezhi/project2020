import os,glob,cv2
from keras.preprocessing.image import img_to_array,array_to_img,load_img,save_img

data = "classdata"
bg = "BG"
bg1 = "BG\RANDOM\IMG_20200503_165351.jpg"

for file  in glob.glob(data + "/*"):
    for img in glob.glob(f"{file}/*"):
        name = img.split("\\")[-1]
        im = img_to_array(load_img(img)) / 255
        im.resize((600,600,3))
        ib = img_to_array(load_img(bg1)) / 255
        ib.resize((600, 600, 3))
        l = (0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9)
        for i in l:
            mix = im*(1-i) + ib * i
            save_img(f"aug/{name}{i}.jpg",array_to_img(mix))
        break
