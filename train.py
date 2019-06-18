from fastai.vision import *

path = 'data/captures'

fnames = list((path / 'train').glob('*img*'))
lbl_names = list((path / 'train').glob('*layer*'))

img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5, 5))
