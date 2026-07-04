import torch
import torch.nn.functional as F

from pcld.utils.consts import PainterConsts, NUM_OF_HYPHENS
from pcld.painter.painter import ActorResNet, RendererFCN


# --- 1. Vectorized Utility Functions ---

def large2small(x: torch.Tensor, divide: int, width: int,
                canvas_cnt: int) -> torch.Tensor:
    """Splits a full-resolution image into a grid of square patches.

    Rearranges the spatial dimensions of a batched image tensor into a flat
    list of (divide × divide) patches, each of size width × width.

    Args:
        x: Full-resolution image batch of shape (B, C, H_large, W_large),
            where H_large == W_large == divide * width.
        divide: Number of patches per spatial dimension.
        width: Side length of each patch in pixels.
        canvas_cnt: Total number of patches per image (divide * divide).

    Returns:
        Patch tensor of shape (B * canvas_cnt, C, width, width).
    """
    B, C, H, W = x.shape
    x = x.view(B, C, divide, width, divide, width)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B * canvas_cnt, C, width, width)
    return x


def small2large(x: torch.Tensor, B: int, divide: int, width: int) -> torch.Tensor:
    """Reassembles a grid of patches into a single full-resolution image.

    Reverses the operation of large2small, merging (divide × divide) patches
    back into a contiguous spatial layout.

    Args:
        x: Patch tensor of shape (B * divide * divide, C, width, width).
        B: Original batch size before patching.
        divide: Number of patches per spatial dimension.
        width: Side length of each patch in pixels.

    Returns:
        Full-resolution tensor of shape (B, C, divide * width, divide * width).
    """
    C = x.shape[1]
    x = x.view(B, divide, divide, C, width, width)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, divide * width, divide * width)
    return x


def smooth(img: torch.Tensor) -> torch.Tensor:
    """Applies a 3×3 box blur using depthwise convolution, preserving borders.

    Smooths interior pixels with a uniform 3×3 kernel while leaving the one-
    pixel border identical to the input, preventing edge artefacts after
    patch reassembly.

    Args:
        img: Image batch of shape (B, 3, H, W) on any device.

    Returns:
        Smoothed image batch of the same shape, with border pixels unchanged.
    """
    kernel = torch.ones((3, 1, 3, 3), device=img.device) / 9.0
    img_padded = F.pad(img, (1, 1, 1, 1), mode='replicate')
    smoothed = F.conv2d(img_padded, kernel, groups=3)

    smoothed[:, :, 0, :] = img[:, :, 0, :]
    smoothed[:, :, -1, :] = img[:, :, -1, :]
    smoothed[:, :, :, 0] = img[:, :, :, 0]
    smoothed[:, :, :, -1] = img[:, :, :, -1]
    return smoothed


def decode(x: torch.Tensor, canvas: torch.Tensor, decoder: RendererFCN,
           width: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Renders a bundle of 5 strokes onto the current canvas.

    Splits the 65-dimensional action vector into 5 groups of 13 parameters
    (10 geometry + 3 colour), renders each stroke alpha mask, composites each
    stroke onto the canvas in order, and collects intermediate canvases.

    Args:
        x: Stroke action bundle of shape (B, 65) where B includes all patches.
        canvas: Current canvas state of shape (B, 3, width, width).
        decoder: RendererFCN that maps (B, 10) geometry params to alpha masks.
        width: Canvas width/height in pixels.

    Returns:
        Tuple of (final_canvas, intermediate_canvases) where:
            final_canvas: Canvas after all 5 strokes, shape (B, 3, width, width).
            intermediate_canvases: List of 5 canvas tensors, one after each
                stroke, each of shape (B, 3, width, width).
    """
    x = x.view(-1, 13)
    stroke = 1.0 - decoder(x[:, :10])
    stroke = stroke.view(-1, 1, width, width)
    color_stroke = stroke * x[:, -3:].view(-1, 3, 1, 1)

    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)

    res = []
    inv_stroke = 1.0 - stroke
    for i in range(5):
        canvas = canvas * inv_stroke[:, i] + color_stroke[:, i]
        res.append(canvas)

    return canvas, res


def prepare_output(canvas: torch.Tensor, to_shape: int, divide: int,
                   width: int, B: int,
                   is_divide: bool = False) -> torch.Tensor:
    """Post-processes a canvas into a uint8 HWC tensor at the target resolution.

    Optionally merges patches back into a full image and smooths the seams,
    then bilinearly resizes to `to_shape × to_shape` and converts to uint8.

    Args:
        canvas: Current canvas, either (B, 3, width, width) for Phase 1 or
            (B * divide * divide, 3, width, width) for Phase 2.
        to_shape: Target spatial resolution (square) in pixels.
        divide: Number of patches per spatial dimension (used in Phase 2).
        width: Patch width/height in pixels.
        B: Original batch size (needed for patch reassembly in Phase 2).
        is_divide: True when `canvas` is in patch layout (Phase 2 output).

    Returns:
        Uint8 tensor of shape (B, to_shape, to_shape, 3) with values in [0, 255].
    """
    output = canvas.detach()

    if is_divide:
        output = small2large(output, B, divide, width)
        output = smooth(output)

    output = F.interpolate(output, size=(to_shape, to_shape), mode='bilinear',
                           align_corners=False)
    output = (output * 255).clamp(0, 255).to(torch.uint8)
    return output.permute(0, 2, 3, 1)


# --- 2. Main Painting Logic ---

def paint(img: torch.Tensor, output_every: list[int], device: str,
          actor: ActorResNet, renderer: RendererFCN) -> torch.Tensor:
    """Paints a batch of images using the neural stroke-based renderer.

    Runs a two-phase painting loop:
    - Phase 1 (global): MAX_STEP/2 steps on a single WIDTH×WIDTH canvas.
    - Phase 2 (patched): MAX_STEP/2 steps on a divide×divide grid of patches,
      which is then merged back into a full-resolution image.

    Canvas snapshots are saved at every stroke count in `output_every`.

    Args:
        img: Input image batch in any of:
            - numpy array (H, W, 3) or (B, H, W, 3)
            - torch Tensor (H, W, 3), (C, H, W), or (B, C, H, W)
            Values may be in [0, 1] or [0, 255] (auto-normalised).
        output_every: Ordered list of stroke-count checkpoints at which to
            capture canvas snapshots.
        device: Target device string (e.g. 'cuda').
        actor: ActorResNet that predicts stroke parameters.
        renderer: RendererFCN that rasterises individual strokes.

    Returns:
        Float tensor of shape (B, Steps, 3, H, W) in [0, 1], where Steps is
        len(output_every). Returns an empty tensor if no checkpoints are hit.
    """
    max_step = PainterConsts.MAX_STEP
    canvas_cnt = PainterConsts.DIVIDE * PainterConsts.DIVIDE
    output_every = set(output_every)

    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img).to(device).float()
    else:
        img = img.to(device).float()

    if img.ndim == 3:
        if img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
    if img.max() > 1.0:
        img = img / 255.0

    B = img.shape[0]
    output_width = img.shape[-1]

    patch_img_full = F.interpolate(img, size=(PainterConsts.WIDTH * PainterConsts.DIVIDE,
                                              PainterConsts.WIDTH * PainterConsts.DIVIDE),
                                   mode='bilinear')
    patch_img = large2small(patch_img_full, PainterConsts.DIVIDE, PainterConsts.WIDTH, canvas_cnt)

    img_low = F.interpolate(img, size=(PainterConsts.WIDTH, PainterConsts.WIDTH), mode='bilinear')

    i_grid = torch.linspace(0, 1, PainterConsts.WIDTH, device=device).view(-1, 1).repeat(1, PainterConsts.WIDTH)
    j_grid = torch.linspace(0, 1, PainterConsts.WIDTH, device=device).view(1, -1).repeat(PainterConsts.WIDTH, 1)
    coord = torch.stack([i_grid, j_grid], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    T = torch.ones([B, 1, PainterConsts.WIDTH, PainterConsts.WIDTH], device=device)
    canvas = torch.zeros([B, 3, PainterConsts.WIDTH, PainterConsts.WIDTH], device=device)

    if PainterConsts.DIVIDE > 1:
        max_step //= 2

    img_idx = 0
    output_canvases = []

    with torch.no_grad():
        # PHASE 1: Global pass
        for i in range(max_step):
            stepnum = T * (i / max_step)
            actor_input = torch.cat([canvas, img_low, stepnum, coord], 1)
            actions = actor(actor_input)
            canvas, res = decode(actions, canvas, renderer, PainterConsts.WIDTH)

            for j in range(5):
                img_idx += 1
                if img_idx in output_every:
                    out = prepare_output(res[j], output_width, PainterConsts.DIVIDE,
                                         PainterConsts.WIDTH, B, False)
                    output_canvases.append(out)

        # PHASE 2: Patched (local) pass
        if PainterConsts.DIVIDE > 1:
            canvas = F.interpolate(canvas, size=(PainterConsts.WIDTH * PainterConsts.DIVIDE,
                                                  PainterConsts.WIDTH * PainterConsts.DIVIDE),
                                   mode='bilinear')
            canvas = large2small(canvas, PainterConsts.DIVIDE, PainterConsts.WIDTH, canvas_cnt)

            coord_p = coord.unsqueeze(1).expand(-1, canvas_cnt, -1, -1, -1).reshape(
                B * canvas_cnt, 2, PainterConsts.WIDTH, PainterConsts.WIDTH)
            T_p = T.unsqueeze(1).expand(-1, canvas_cnt, -1, -1, -1).reshape(
                B * canvas_cnt, 1, PainterConsts.WIDTH, PainterConsts.WIDTH)

            for i in range(max_step):
                stepnum = T_p * (i / max_step)
                actor_input = torch.cat([canvas, patch_img, stepnum, coord_p], 1)
                actions = actor(actor_input)
                canvas, res = decode(actions, canvas, renderer, PainterConsts.WIDTH)

                for j in range(5):
                    img_idx += canvas_cnt
                    if img_idx in output_every:
                        out = prepare_output(res[j], output_width, PainterConsts.DIVIDE,
                                              PainterConsts.WIDTH, B, True)
                        output_canvases.append(out)

    if not output_canvases:
        return torch.empty(0)

    final_out = torch.stack(output_canvases, dim=0).float() / 255.0
    final_out = final_out.permute(1, 0, 4, 2, 3)

    return final_out


def paint_images(x: torch.Tensor, output_every: list[int], device: str,
                 actor: ActorResNet, renderer: RendererFCN,
                 add_original: bool = True) -> torch.Tensor:
    """Paints a batch of images and optionally appends the originals as a step.

    Calls `paint` to generate canvas snapshots, then concatenates the original
    (unmodified) input along the step dimension so the PCLD classifier also
    sees the clean image as if it were painted at t=∞.

    Args:
        x: Input image batch of shape (B, 3, H, W) in [0, 1].
        output_every: Stroke-count checkpoints for canvas snapshots.
        device: Target device string.
        actor: ActorResNet stroke-parameter predictor.
        renderer: RendererFCN stroke renderer.
        add_original: If True, appends the original image as the final step,
            making the output shape (B, Steps + 1, 3, H, W).

    Returns:
        Float tensor of shape (B, Steps[+1], 3, H, W) in [0, 1].
    """
    canvases = paint(x, output_every, device, actor, renderer).to(device)
    if add_original:
        orig = x.unsqueeze(1).to(device)
        canvases = torch.cat([canvases, orig], dim=1)
    return canvases


def load_painter(actor_path: str, renderer_path: str,
                 device: str) -> tuple[ActorResNet, RendererFCN]:
    """Loads the pretrained actor and renderer models from disk.

    Args:
        actor_path: Filesystem path to the actor checkpoint (.pkl or .pth).
        renderer_path: Filesystem path to the renderer checkpoint (.pkl or .pth).
        device: Target device string (e.g. 'cuda' or 'cpu').

    Returns:
        Tuple of (actor, renderer), both in eval mode on `device`.
    """
    print('-' * NUM_OF_HYPHENS)
    print('Loading painter...')

    actor = ActorResNet()
    actor.load_state_dict(torch.load(actor_path))
    renderer = RendererFCN()
    renderer.load_state_dict(torch.load(renderer_path))

    actor = actor.to(device).eval()
    renderer = renderer.to(device).eval()
    print('Finished loading painter!')

    return actor, renderer
