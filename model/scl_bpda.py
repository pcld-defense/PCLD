import torch
import torch.nn as nn
torch.manual_seed(42)

class BPDASketcherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, non_diff_layer, grad_approx_net, device):
        ctx.save_for_backward(input)
        ctx.grad_approx_net = grad_approx_net
        output = non_diff_layer(input, device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_approx_net = ctx.grad_approx_net
        grad_approx_net.eval()  # Ensure it is in eval mode
        with torch.no_grad():
            approx_grad = grad_approx_net(input)
        new_grad_input = grad_output * approx_grad
        return new_grad_input, None, None, None  # None for the non_diff_layer, grad_approx_net, device

class BPDASketcher(nn.Module):
    def __init__(self, non_diff_layer, grad_approx_net, device):
        super(BPDASketcher, self).__init__()
        self.non_diff_layer = non_diff_layer
        self.grad_approx_net = grad_approx_net
        self.device = device

    def forward(self, x):
        return BPDASketcherLayer.apply(x,
                                       self.non_diff_layer,
                                       self.grad_approx_net,
                                       self.device)


class SCL(nn.Module):
    def __init__(self, sketcher, clf):
        super(SCL, self).__init__()
        self.sketcher = sketcher
        self.clf = clf

    def forward(self, x):
        x = self.sketcher(x)
        # x = x.view(torch.Size([-1]) + x.size()[2:])
        x = self.clf(x)
        # x = torch.softmax(x, dim=1)
        return x
