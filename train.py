from fastai.vision import *


def just_image(x):
    return 'img' in str(x)


def acc_segmentation(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


if __name__ == "__main__":
    path = Path('data/captures')
    print(path.ls())

    fnames = list((path / 'train').glob('*img*'))
    print(fnames[:3])
    lbl_names = list((path / 'train').glob('*layer*'))
    print(lbl_names[:3])

    img_f = fnames[0]
    img = open_image(img_f)
    img.show(figsize=(5, 5))

    get_y_fn = lambda x: str(x).replace('img', 'layer')
    open_image(get_y_fn(img_f), convert_mode='L').data.unique()

    mask = open_mask(get_y_fn(img_f))
    mask.show(figsize=(5, 5))

    src_size = np.array(mask.shape[1:])
    print(src_size, mask.data)

    codes = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "Cube", "Sphere", "Sphere"])

    size = src_size // 2
    bs = 8

    src = (SegmentationItemList.from_folder(path)
           .filter_by_func(just_image)
           .split_by_folder(train='train', valid='val')
           .label_from_func(get_y_fn, classes=codes))
    print(src)

    data = (src.transform(get_transforms(), size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(imagenet_stats))
    data.show_batch(2, figsize=(10, 7))
    data.show_batch(2, figsize=(10, 7), ds_type=DatasetType.Valid)

    name2id = {v: k for k, v in enumerate(codes)}
    void_code = name2id['0']

    metrics = acc_segmentation
    wd = 1e-2
    learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

    lr_find(learn)
    learn.recorder.plot()

    lr = 1e-3
    learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
    learn.save('stage-1')
    learn.load('stage-1')
    learn.show_results(rows=3, figsize=(16, 16))
    img = open_image((path / 'val').ls()[0])
    display(img)
    plt.imshow(learn.predict(img)[1].squeeze())

    plt.show()
