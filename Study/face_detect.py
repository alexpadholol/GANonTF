import cv2
import os

# Get user supplied values
# Create the haar cascade

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

#read the image
for fn in sorted(os.listdir('img_align_celeba')):
    #print(fn)
    image = cv2.imread('img_align_celeba/' + fn)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,5,5)
    if len(faces) == 0:
        pass
    else:
        x,y,w,h = faces[0]
        image_corp = image[y:y+w,x:x+w,:]
        image_resize = cv2.resize(image_corp,(64,64))
        #print(image_resize)
        cwd = os.getcwd()
        print(cwd+'/64_corp/' + fn[:-4] + '_corp' + fn[-4:])
        cv2.imwrite('F:\github\gan\GAN\GANonTF\Study\\64_corp' + '/' +  fn[:-4] + '_corp' + fn[-4:],image_resize)