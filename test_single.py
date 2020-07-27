from __future__ import print_function, division
from skimage import io, color
import torch
import numpy as np
from xception import Xception

class ToTensor(object):
    def __call__(self, im):

        gray_image = color.rgb2gray(im)
        gray_image = np.expand_dims(gray_image, -1)
        gray_image = gray_image.transpose((2, 0, 1))
        gray_image = torch.from_numpy(gray_image).view(1,1,128,128)
        print(gray_image)
        return gray_image


class DeepfakeTest():
    def __init__(self, model, ckpt_dir):
        self.transform = ToTensor()
        self.checkpoint = torch.load(ckpt_dir)
        self.model = model
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        # Model on cuda
        if torch.cuda.is_available():
            model = model.cuda()

    def test_im(self, im_dir):
        im = self.transform(io.imread(im_dir))
        with torch.no_grad():
            input_image = im.cuda().float()
            # compute output
            output = self.model(input_image).cpu().numpy()
        return output

if __name__ == '__main__':
    tester = DeepfakeTest(model=Xception(), ckpt_dir='log_path/Xception_trained_model.pth')
    imgs = ['sample/trump.png']
    results = [tester.test_im(img) for img in imgs]
    print(results)

