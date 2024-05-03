import torch
import numpy as np
import torch
import torchvision
from PIL import Image
import zipfile
import argparse
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.utils as vtils

#from util.eval_utils import build_resizer
#from util.utils import unsymmetrize_image_data, symmetrize_image_data, random_seed

def prepare_image(img,size):
    img = np.array(img).astype('float')
    img = (img - 127.5)/127.5
    minibatch = torch.tensor(img).unsqueeze_(0)
    minibatch = minibatch.permute(0,3,1,2)

    minibatch = torch.nn.functional.interpolate(minibatch, size = (size, size), mode = 'bilinear')
    minibatch = ((minibatch + 1) / 2)
    minibatch = minibatch.numpy().transpose((0,2,3,1))
    minibatch = (minibatch*255).astype('uint8')
    return minibatch[0]

def build_resizer(mode, size):
    if mode == "clean":
        return make_resizer("PIL", False, "bicubic", (size, size))
    # if using legacy tensorflow, do not manually resize outside the network
    elif mode == "legacy_tensorflow":
        return lambda x: x
    elif mode == "legacy_pytorch":
        return make_resizer("PyTorch", False, "bilinear", (299, 299))
    else:
        raise ValueError(f"Invalid mode {mode} specified")

"""
Construct a function that resizes a numpy image based on the
flags passed in.
"""
def make_resizer(library, quantize_after, filter, output_size):
    if library == "PIL" and quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX
        }
        def func(x):
            x = Image.fromarray(x)
            x = x.resize(output_size, resample=name_to_filter[filter])
            x = np.asarray(x).clip(0, 255).astype(np.uint8)
            return x
    elif library == "PIL" and not quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX
        }
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=name_to_filter[filter])
            return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    elif library == "TensorFlow":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        import tensorflow as tf
        def func(x):
            x = tf.constant(x)[tf.newaxis, ...]
            x = tf.image.resize(x, output_size, method=filter)
            x = x[0, ...].numpy().clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    elif library == "OpenCV":
        import cv2
        name_to_filter = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA
        }
        def func(x):
            x = cv2.resize(x, output_size, interpolation=name_to_filter[filter])
            x = x.clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores
    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=256, fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode, size = size)
        self.custom_image_tranform = lambda x: x
        self._zipfile = None

    def _get_zipfile(self):
        assert self.fdir is not None and '.zip' in self.fdir
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.fdir)
        return self._zipfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if self.fdir is not None and '.zip' in self.fdir:
            with self._get_zipfile().open(path, 'r') as f:
                img_np = np.array(Image.open(f).convert('RGB'))
        elif ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil)

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy', 'JPEG', 'JPG', 'PNG'}


def main(args):
    # Checkpoint
    resize = build_resizer(mode = 'clean', size = args.size)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = torchvision.datasets.ImageFolder(args.pth, transform= train_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.bs,
        shuffle = False, pin_memory=False, num_workers=2, drop_last=False)

    length = 0
    for idx, (x, y) in enumerate(train_queue):
        length += x.shape[0]
        x = x*2 -1
        #transform
        x = torchvision.transforms.functional.resize(x, 256, antialias = True)
        x = x.clamp(-1., 1.)
        x = (x+1)/2.
        for i in range(x.shape[0]):
            img = x[i].data.cpu().numpy().transpose((1,2,0))
            img = Image.fromarray((img * 255).astype(np.uint8))
            img.save('/hub_data2/dogyun/iclr_data/real_256_dog/256-{}-{}.jpg'.format(idx, i))
            #img.save('/hub_data2/dogyun/data/afhqdog_512_real_png/img-{}-{}.png'.format(idx, i), compress_level = 0, optimize = False)
            #img.save('/hub_data/dogyun/afhq256_real_png3/img-{}-{}.png'.format(idx, i), optimize = True)
            #vtils.save_image(x[i], './ffhq_real_256_2/img-{}-{}.png'.format(idx, i))
        #if length >= 1000:
        #    break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test')
    #parser.add_argument('--ngpu', type = str, default='0')
    parser.add_argument('--bs', type = int, default = 50)
    parser.add_argument('--pth', type = str, default='/mnt/prj/')
    parser.add_argument('--size', type = int, default= 256)
    args = parser.parse_args()

    main(args)