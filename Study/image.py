class save_img(object):
    def save_images(self,img,path):
        from PIL import Image

        im = Image.fromarray(img)
        im.save(path)
