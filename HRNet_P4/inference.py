from dis import Instruction
from multiprocessing.resource_sharer import stop
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from misc.utils import get_final_preds

from HRnet import HRNet

class SimpleHRNet:
    def __init__(self,
                 c,
                 key,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(256, 192),
                 interpolation=cv2.INTER_CUBIC,
                 multiperson=True,
                 return_heatmaps=False,
                 return_bounding_boxes=False,
                 max_batch_size=32,
                 device=torch.device("cpu")):

        self.c = c
        self.key = key
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.device = device

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, key=key)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
                transforms.ToTensor()
        ])

    # Image is passed from cv2
    def predict_single(self, image):
        old_res = image.shape
        scale = (old_res[1], old_res[0])

        # Transformation   
        image = cv2.resize(
            image,
            (self.resolution[1], self.resolution[0]),
        interpolation=self.interpolation
        )
        image = self.transform(image).to(self.device)

        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)

        # Model Prediction
        self.model.eval()
        out = self.model(image)
        preds, maxvals = get_final_preds(out, scale)

        return preds, maxvals

if __name__ == '__main__':
    import torch
    from torch import nn
    import cv2
    import os

    simplehrnet = SimpleHRNet(c=48, key=12, checkpoint_path='./logs/20221123_0906/checkpoint_best_loss.pth', device=torch.device('cuda'))
    Instruction = True

    for image in os.listdir('./datasets/COCO/default_val'):
        if Instruction == True:
            image = cv2.imread(os.path.join('./datasets/COCO/default_val', image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out = simplehrnet.predict_single(image)
            
            # Out: Batch, Points, XY

            for pair in out[0][0]:
                image = cv2.circle(image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
        
            cv2.imshow('Image', cv2.resize(image, (1000, 1000)))
            cv2.waitKey(1000)