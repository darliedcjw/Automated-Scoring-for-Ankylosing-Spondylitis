import torch.nn as nn
import math

class focal_loss(nn.Module):
    def __init__(self, 
        wf=0.1,
        fp=2):

        self.wf = wf
        self.fp = fp

    def forward(self, p, index):
        loss = 0
        preds = list(zip(p.tolist(), index.tolist()))
        for p, index in preds:
            if index == 0:
                loss += self.wf*(1 - 0.8)**self.fp*math.log(p)
            elif index == 1:
                loss += self.wf*(1 - 0.8)**self.fp*math.log(1 - p)
        return loss
        

