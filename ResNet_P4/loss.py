import torch
import torch.nn as nn

class focal_loss(nn.Module):
    def __init__(self, 
        wf=0.1,
        fp=2):
        super().__init__()

        self.wf = wf
        self.fp = fp

    def forward(self, outputs, labels):
        outputs = nn.Softmax(dim=1)(outputs)
        for index, output in enumerate(outputs):
            loss = 0
            class_idx = labels[index]
            p = output[class_idx]
            if class_idx == 0:
                loss += -self.wf*(1 - p)**self.fp*torch.log(p)
            elif class_idx == 1:
                loss += -(1-self.wf)*(1 - p)**self.fp*torch.log(1 - p)
        return loss
        

