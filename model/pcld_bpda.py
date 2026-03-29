from typing import Optional, Union

import torch
import torch.nn as nn
from cleverhans.torch.utils import optimize_linear

from painter.painter_surrogate import PainterSurrogate

torch.manual_seed(42)


class BPDAPainterLayer(torch.autograd.Function):
    """Custom autograd function implementing BPDA for the non-differentiable painter.

    During the forward pass the real (non-differentiable) painter is called to
    produce canvases. During the backward pass a differentiable surrogate painter
    is used to compute a Jacobian-vector product (JVP) that approximates the true
    painter gradient, enabling gradient-based attacks to propagate through the
    non-differentiable rendering step (Athalye et al., ICML 2018).

    Class attributes cache state across forward/backward calls within a single
    attack iteration:
        _stored_non_diff_layer: Cached canvas tensor (canvas-level attack variant).
        _stored_grad_output: Upstream gradient stored for the canvas-level
            perturbation step.
    """

    _stored_non_diff_layer = None
    _stored_grad_output = None

    @classmethod
    def reset(cls) -> None:
        """Clears all cached state.

        Must be called at the start of each new batch and at the start of each
        random restart to prevent stale canvases from contaminating new iterates.
        """
        cls._stored_non_diff_layer = None
        cls._stored_grad_output = None

    @staticmethod
    def forward(ctx, input: torch.Tensor, non_diff_layer, grad_approx_net,
                output_every: list[int], device: str, actor: nn.Module,
                renderer: nn.Module, epsilon: Union[float, None] = None,
                norm: Union[str, None] = None,
                painter_config=None) -> torch.Tensor:
        """Runs the painter forward and saves context for the backward pass.

        Args:
            ctx: Autograd context object for saving tensors.
            input: Adversarial input image batch of shape (B, 3, H, W).
            non_diff_layer: Either a PainterSurrogate (used as a fast
                differentiable approximation), a pre-computed canvas tensor
                (used to skip repainting), or the real paint_images callable.
            grad_approx_net: The surrogate model used in the backward pass to
                approximate the painter Jacobian via a JVP.
            output_every: Stroke-count checkpoints at which canvases are saved.
            device: Target device string (e.g. 'cuda').
            actor: ActorResNet stroke-parameter predictor.
            renderer: RendererFCN stroke renderer.
            epsilon: L-inf perturbation budget for canvas-level perturbation
                (only used when non_diff_layer is a Tensor).
            norm: Attack norm type (e.g. 'inf') for canvas-level perturbation.
            painter_config: Optional ``PainterConfig`` passed through to
                ``paint_images`` for native-resolution painting.

        Returns:
            Canvas tensor of shape (B, Steps, 3, H, W) in [0, 1].
        """
        ctx.save_for_backward(input)
        ctx.grad_approx_net = grad_approx_net
        if type(non_diff_layer) == PainterSurrogate:
            output = non_diff_layer(input)
        elif isinstance(non_diff_layer, torch.Tensor):
            if BPDAPainterLayer._stored_non_diff_layer is None:
                BPDAPainterLayer._stored_non_diff_layer = non_diff_layer
            if BPDAPainterLayer._stored_grad_output is not None:
                optimal_perturbation = optimize_linear(
                    BPDAPainterLayer._stored_grad_output, epsilon, norm=norm)
                BPDAPainterLayer._stored_non_diff_layer += optimal_perturbation
                BPDAPainterLayer._stored_non_diff_layer = torch.clamp(
                    BPDAPainterLayer._stored_non_diff_layer, 0, 1)
            output = BPDAPainterLayer._stored_non_diff_layer
            ctx.grad_approx_net = output
        else:
            output = non_diff_layer(input, output_every, device, actor, renderer,
                                    config=painter_config)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Approximates the painter gradient via a surrogate Jacobian-vector product.

        Implements BPDA (Athalye et al. 2018): replaces the non-differentiable
        painter's true Jacobian with the surrogate's Jacobian during backprop.
        The JVP ``d(surrogate(x))/dx · grad_output`` is computed via
        ``torch.autograd.grad``, giving a proper chain-rule approximation of
        ``dL/dx ≈ dL/d(canvas) · d(surrogate)/dx``.

        Args:
            grad_output: Upstream gradient of shape (B, Steps, 3, H, W).

        Returns:
            Tuple of gradients matching the forward inputs. Only the gradient
            w.r.t. ``input`` is non-None; all others are None.
        """
        input, = ctx.saved_tensors
        grad_approx_net = ctx.grad_approx_net
        BPDAPainterLayer._stored_grad_output = grad_output

        if isinstance(grad_approx_net, torch.Tensor):
            # Canvas-level attack: no surrogate network; gradient approximation
            # degenerates to the stored canvas tensor (legacy path).
            new_grad_input = grad_output.mean(dim=1)
        else:
            # Standard BPDA: compute JVP through the surrogate.
            # Re-run the surrogate with a fresh requires_grad leaf so autograd
            # can differentiate through it, then chain grad_output via autograd.grad.
            grad_approx_net.eval()
            x_for_grad = input.detach().requires_grad_(True)
            with torch.enable_grad():
                surrogate_out = grad_approx_net(x_for_grad)  # (B, Steps, 3, H, W)
            new_grad_input = torch.autograd.grad(
                surrogate_out, x_for_grad,
                grad_outputs=grad_output,
                retain_graph=False,
            )[0]

        return new_grad_input, None, None, None, None, None, None, None, None, None


class BPDAPainter(nn.Module):
    """nn.Module wrapper around BPDAPainterLayer for use inside larger pipelines.

    Holds all painting configuration and delegates to the custom autograd
    function so that the painter can be composed with differentiable modules
    (classifier, decisioner) and attacked end-to-end.
    """

    def __init__(self, non_diff_layer, grad_approx_net: nn.Module,
                 output_every: list[int], device: str, actor: nn.Module,
                 renderer: nn.Module, epsilon: Union[float, None] = None,
                 norm: Union[str, None] = None,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 painter_config=None) -> None:
        """Stores painting configuration.

        Args:
            non_diff_layer: Real painter callable or cached canvas tensor.
            grad_approx_net: Surrogate network used in the backward pass.
            output_every: Stroke-count checkpoints for canvas snapshots.
            device: Target device string.
            actor: ActorResNet stroke-parameter predictor.
            renderer: RendererFCN stroke renderer.
            epsilon: L-inf budget for canvas-level perturbation (optional).
            norm: Attack norm for canvas-level perturbation (optional).
            mean: Per-channel normalisation mean of shape (3,) used by the
                DataLoader transform. When provided, inputs are unnormalised
                to [0, 1] before painting and the painted canvases are
                re-normalised before being passed to the classifier.
            std: Per-channel normalisation std of shape (3,) paired with
                ``mean``. Must be provided together with ``mean``.
            painter_config: Optional ``PainterConfig`` for native-resolution
                painting.  Passed through to ``paint_images``.
        """
        super(BPDAPainter, self).__init__()
        self.non_diff_layer = non_diff_layer
        self.grad_approx_net = grad_approx_net
        self.output_every = output_every
        self.device = device
        self.actor = actor
        self.renderer = renderer
        self.epsilon = epsilon
        self.norm = norm
        self.mean = mean  # (3,) or None
        self.std = std    # (3,) or None
        self.painter_config = painter_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paints the input batch via BPDAPainterLayer.

        If ``mean``/``std`` were supplied, the normalised input is first
        converted back to [0, 1] before painting, and all output canvases
        (including the appended original step) are re-normalised so the
        downstream classifier always receives correctly scaled inputs.

        Args:
            x: Input image batch of shape (B, 3, H, W), normalised when
                ``mean``/``std`` are set.

        Returns:
            Canvas tensor of shape (B, Steps, 3, H, W), normalised when
            ``mean``/``std`` are set, in [0, 1] otherwise.
        """
        if self.mean is not None:
            mean4d = self.mean.to(x.device).view(1, 3, 1, 1)
            std4d  = self.std.to(x.device).view(1, 3, 1, 1)
            x_input = x * std4d + mean4d          # unnormalise → [0, 1]
        else:
            x_input = x

        canvases = BPDAPainterLayer.apply(x_input,
                                          self.non_diff_layer,
                                          self.grad_approx_net,
                                          self.output_every,
                                          self.device,
                                          self.actor,
                                          self.renderer,
                                          self.epsilon,
                                          self.norm,
                                          self.painter_config)

        if self.mean is not None:
            mean5d = mean4d.unsqueeze(1)           # (1, 1, 3, 1, 1)
            std5d  = std4d.unsqueeze(1)
            canvases = (canvases - mean5d) / std5d  # re-normalise for classifier

        return canvases


class PCL(nn.Module):
    """Painter–Classifier pipeline without a decisioner.

    Used during the attack_pcl stage to generate per-step confidence
    trajectories that become the decisioner's training data.
    """

    def __init__(self, painter: nn.Module, clf: nn.Module) -> None:
        """Composes the painter and classifier.

        Args:
            painter: BPDAPainter that returns (B, Steps, 3, H, W) canvases.
            clf: Classifier that maps (B, 3, H, W) → (B, num_classes) logits.
        """
        super(PCL, self).__init__()
        self.painter = painter
        self.clf = clf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paints the input, flattens steps into the batch dimension, classifies.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            Logits of shape (B * Steps, num_classes).
        """
        x = self.painter(x)
        x = x.view(torch.Size([-1]) + x.size()[2:])
        x = self.clf(x)
        return x


class PCLD(nn.Module):
    """Full Painter–Classifier–Decisioner pipeline with BPDA support.

    Paints the input at multiple stroke counts, classifies each canvas,
    aggregates the softmax trajectories, and lets the decisioner produce a
    final class prediction.
    """

    def __init__(self, painter: nn.Module, clf: nn.Module,
                 decisioner: nn.Module, num_paint_steps: int,
                 decisioner_architecture: str = 'conv') -> None:
        """Composes all three pipeline components.

        Args:
            painter: BPDAPainter returning (B, Steps, 3, H, W) canvases.
            clf: Classifier returning (B * Steps, num_classes) logits.
            decisioner: Decisioner1DConv or DecisionerFC that produces final
                class logits from the confidence trajectory.
            num_paint_steps: Total number of paint steps (including t=∞).
            decisioner_architecture: Layout of the decisioner input; 'conv'
                keeps the (B, Steps, num_classes) shape, 'fc' flattens it.
        """
        super(PCLD, self).__init__()
        self.painter = painter
        self.clf = clf
        self.decisioner = decisioner
        self.num_paint_steps = num_paint_steps
        self.decisioner_architecture = decisioner_architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the full PCLD pipeline end-to-end.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            Decisioner logits of shape (B, num_classes).
        """
        x = self.painter(x)
        x = x.view(torch.Size([-1]) + x.size()[2:])
        x = self.clf(x)
        x = torch.softmax(x, dim=1)
        if self.decisioner_architecture == 'conv':
            x = x.view(-1, self.num_paint_steps, x.shape[-1])
        else:
            x = x.reshape(-1, self.num_paint_steps * x.shape[-1])
        x = self.decisioner(x)
        return x


class CLD(nn.Module):
    """Classifier–Decisioner pipeline operating on pre-painted canvases.

    Used for naïve (non-adaptive) baseline attacks where the input is already
    a stack of painted canvases rather than a raw image.
    """

    def __init__(self, clf: nn.Module, decisioner: nn.Module,
                 num_paint_steps: int,
                 decisioner_architecture: str = 'conv') -> None:
        """Composes the classifier and decisioner without a painter.

        Args:
            clf: Classifier returning (B * Steps, num_classes) logits.
            decisioner: Decisioner1DConv or DecisionerFC.
            num_paint_steps: Total number of paint steps fed to the decisioner.
            decisioner_architecture: 'conv' or 'fc' input layout for the
                decisioner.
        """
        super(CLD, self).__init__()
        self.clf = clf
        self.decisioner = decisioner
        self.num_paint_steps = num_paint_steps
        self.decisioner_architecture = decisioner_architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies pre-painted canvases and feeds trajectories to decisioner.

        Args:
            x: Pre-painted canvas batch of shape (B, Steps, 3, H, W).

        Returns:
            Decisioner logits of shape (B, num_classes).
        """
        x = x.view(torch.Size([-1]) + x.size()[2:])
        x = self.clf(x)
        x = torch.softmax(x, dim=1)
        if self.decisioner_architecture == 'conv':
            x = x.view(-1, self.num_paint_steps, x.shape[-1])
        else:
            x = x.reshape(-1, self.num_paint_steps * x.shape[-1])
        x = self.decisioner(x)
        return x
