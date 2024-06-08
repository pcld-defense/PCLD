import torch
import torch.nn as nn
from cleverhans.torch.utils import optimize_linear

from model.painter_surrogate import PainterSurrogate

torch.manual_seed(42)


# class BPDAPainterLayer(torch.autograd.Function):
#     _stored_non_diff_layer = None
#     _stored_grad_output = None
#
#     @staticmethod
#     def forward(ctx, input, non_diff_layer, grad_approx_net, output_every,
#                 device, actor, renderer, epsilon=None, norm=None):
#         ctx.save_for_backward(input)
#         ctx.grad_approx_net = grad_approx_net
#         if type(non_diff_layer) == PainterSurrogate:  # surrogate painter
#             output = non_diff_layer(input)
#
#         elif isinstance(non_diff_layer, torch.Tensor):
#             if BPDAPainterLayer._stored_non_diff_layer is None:
#                 BPDAPainterLayer._stored_non_diff_layer = non_diff_layer
#             if BPDAPainterLayer._stored_grad_output is not None:
#                 optimal_perturbation = optimize_linear(BPDAPainterLayer._stored_grad_output, epsilon, norm=norm)
#                 BPDAPainterLayer._stored_non_diff_layer += optimal_perturbation
#                 BPDAPainterLayer._stored_non_diff_layer = torch.clamp(BPDAPainterLayer._stored_non_diff_layer, 0, 1)
#
#             output = BPDAPainterLayer._stored_non_diff_layer  # Simply use canvases approximation for better time performance
#         else:  # use the non diff painter as is
#             output = non_diff_layer(input, output_every, device, actor, renderer)
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_approx_net = ctx.grad_approx_net
#         grad_approx_net.eval()  # Ensure it is in eval mode
#         with torch.no_grad():
#             approx_grad = grad_approx_net(input)
#         new_grad_input = (grad_output * approx_grad).mean(dim=1)  # avg gradients from all paint steps
#         # Store the grad_output for the next forward pass
#         BPDAPainterLayer._stored_grad_output = grad_output
#         return new_grad_input, None, None, None, None, None, None, \
#                None, None  # None for the non_diff_layer, grad_approx_net, output_every, device,
#         # actor, renderer, epsilon, norm


class BPDAPainterLayer(torch.autograd.Function):
    _stored_non_diff_layer = None
    _stored_grad_output = None

    @staticmethod
    def forward(ctx, input, non_diff_layer, grad_approx_net, output_every,
                device, actor, renderer, epsilon=None, norm=None):
        ctx.save_for_backward(input)
        ctx.grad_approx_net = grad_approx_net
        if type(non_diff_layer) == PainterSurrogate:  # surrogate painter
            output = non_diff_layer(input)

        elif isinstance(non_diff_layer, torch.Tensor):
            if BPDAPainterLayer._stored_non_diff_layer is None:
                BPDAPainterLayer._stored_non_diff_layer = non_diff_layer
            if BPDAPainterLayer._stored_grad_output is not None:
                optimal_perturbation = optimize_linear(BPDAPainterLayer._stored_grad_output, epsilon, norm=norm)
                BPDAPainterLayer._stored_non_diff_layer += optimal_perturbation
                BPDAPainterLayer._stored_non_diff_layer = torch.clamp(BPDAPainterLayer._stored_non_diff_layer, 0, 1)

            output = BPDAPainterLayer._stored_non_diff_layer  # Simply use canvases approximation for better time performance
            ctx.grad_approx_net = output
        else:  # use the non diff painter as is
            output = non_diff_layer(input, output_every, device, actor, renderer)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_approx_net = ctx.grad_approx_net
        if isinstance(grad_approx_net, torch.Tensor):
            approx_grad = grad_approx_net.detach()
        else:
            grad_approx_net.eval()  # Ensure it is in eval mode
            with torch.no_grad():
                approx_grad = grad_approx_net(input)
        new_grad_input = (grad_output * approx_grad).mean(dim=1)  # avg gradients from all paint steps
        # Store the grad_output for the next forward pass
        BPDAPainterLayer._stored_grad_output = grad_output
        return new_grad_input, None, None, None, None, None, None, \
               None, None  # None for the non_diff_layer, grad_approx_net, output_every, device,
        # actor, renderer, epsilon, norm


class BPDAPainter(nn.Module):
    def __init__(self, non_diff_layer, grad_approx_net, output_every,
                 device, actor, renderer, epsilon=None, norm=None):
        super(BPDAPainter, self).__init__()
        self.non_diff_layer = non_diff_layer
        self.grad_approx_net = grad_approx_net
        self.output_every = output_every
        self.device = device
        self.actor = actor
        self.renderer = renderer
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, x):
        return BPDAPainterLayer.apply(x,
                                      self.non_diff_layer,
                                      self.grad_approx_net,
                                      self.output_every,
                                      self.device,
                                      self.actor,
                                      self.renderer,
                                      self.epsilon,
                                      self.norm)


class PCL(nn.Module):
    def __init__(self, painter, clf):
        super(PCL, self).__init__()
        self.painter = painter
        self.clf = clf

    def forward(self, x):
        x = self.painter(x)
        batch_steps = x.size()[:2]
        x = x.view(torch.Size([-1]) + x.size()[2:])
        x = self.clf(x)
        # x = torch.softmax(output, dim=1)
        return x


class PCLD(nn.Module):
    def __init__(self, painter, clf, decisioner, num_paint_steps,
                 decisioner_architecture='conv'):
        super(PCLD, self).__init__()
        self.painter = painter
        self.clf = clf
        self.decisioner = decisioner
        self.num_paint_steps = num_paint_steps
        self.decisioner_architecture = decisioner_architecture

    def forward(self, x):
        x = self.painter(x)
        x = x.view(torch.Size([-1]) + x.size()[2:])
        x = self.clf(x)
        x = torch.softmax(x, dim=1)
        if self.decisioner_architecture == 'conv':
            x = x.view(-1, self.num_paint_steps, x.shape[-1])
        else:  # fc
            x = x.reshape(-1, self.num_paint_steps*x.shape[-1])
        x = self.decisioner(x)
        return x


# ========== CLD model for Naïve attacks ========== #
class CLD(nn.Module):
    def __init__(self, clf, decisioner, num_paint_steps, decisioner_architecture='conv'):
        super(CLD, self).__init__()
        self.clf = clf
        self.decisioner = decisioner
        self.num_paint_steps = num_paint_steps
        self.decisioner_architecture = decisioner_architecture

    def forward(self, x):
        batch_steps = x.size()[:2]
        x = x.view(torch.Size([-1]) + x.size()[2:])
        x = self.clf(x)
        x = torch.softmax(x, dim=1)
        if self.decisioner_architecture == 'conv':
            x = x.view(-1, self.num_paint_steps, x.shape[-1])
        else:  # fc
            x = x.reshape(-1, self.num_paint_steps*x.shape[-1])
        x = self.decisioner(x)
        return x
