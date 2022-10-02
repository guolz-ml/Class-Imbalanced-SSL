import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FixMatch(nn.Module):
    def __init__(self, args, temperature, threshold):
        super().__init__()
        self.device = args.device
        self.mu = args.mu
        self.T = temperature
        self.threshold = threshold

    def forward(self, inputs_uw, inputs_us, model):

        inputs_uw, inputs_us = inputs_uw.to(self.device), inputs_us.to(self.device)
        outputs_u = model(inputs_uw)[0]
        targets_u = torch.softmax(outputs_u, dim=1)

        max_p, p_hat = torch.max(targets_u, dim=1)

        mask = max_p.ge(self.threshold).float()
        outputs = model(inputs_us)[0]

        ssl_loss = (F.cross_entropy(outputs, p_hat, reduction='none') * mask).mean()
        return ssl_loss


class ADSH(nn.Module):
    def __init__(self, args, temperature, threshold):
        super().__init__()
        self.device = args.device
        self.mu = args.mu
        self.T = temperature
        self.threshold = threshold

    def forward(self, inputs_uw, inputs_us, model, score):

        inputs_uw, inputs_us = inputs_uw.to(self.device), inputs_us.to(self.device)
        outputs_uw = model(inputs_uw)[0]
        probs = torch.softmax(outputs_uw, dim=1)

        rectify_prob = probs / torch.from_numpy(score).float().to(self.device)
        max_rp, rp_hat = torch.max(rectify_prob, dim=1)
        mask = max_rp.ge(1.0)

        outputs = model(inputs_us)[0]

        ssl_loss = (F.cross_entropy(outputs, rp_hat, reduction='none') * mask).mean()
        return ssl_loss
