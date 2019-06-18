from fastai.vision import open_image
# from fastai.callbacks.hooks import *

if __name__ == "__main__":
    path = 'data/captures'

    fnames = list((path / 'train').glob('*img*'))
    print(fnames[:3])
    lbl_names = list((path / 'train').glob('*layer*'))
    print(lbl_names[:3])

    img_f = fnames[0]
    img = open_image(img_f)
    img.show(figsize=(5, 5))



