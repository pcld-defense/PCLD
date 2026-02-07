import torch
import torch.nn.functional as F

from util.consts import PainterConsts, NUM_OF_HYPHENS
from model.painter import ActorResNet, RendererFCN


# --- 1. Vectorized Utility Functions ---

def large2small(x, divide, width, canvas_cnt):
    """Split the image into patches using pure Torch."""
    # Input x: (B, C, H_large, W_large)
    B, C, H, W = x.shape
    x = x.view(B, C, divide, width, divide, width)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B * canvas_cnt, C, width, width)
    return x


def small2large(x, divide, width):
    """Combine patches into original image using pure Torch."""
    # Input x: (divide*divide, C, width, width)
    C = x.shape[1]
    x = x.view(divide, divide, C, width, width)
    x = x.permute(2, 0, 3, 1, 4).reshape(C, divide * width, divide * width)
    return x


def smooth(img):
    """Apply 3x3 box blur on GPU using Depthwise Convolution."""
    # img: (3, H, W)
    kernel = torch.ones((3, 1, 3, 3), device=img.device) / 9.0

    # Pad to maintain resolution
    img_padded = F.pad(img.unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    smoothed = F.conv2d(img_padded, kernel, groups=3).squeeze(0)

    # Restore boundary pixels
    smoothed[:, 0, :] = img[:, 0, :]
    smoothed[:, -1, :] = img[:, -1, :]
    smoothed[:, :, 0] = img[:, :, 0]
    smoothed[:, :, -1] = img[:, :, -1]
    return smoothed


def decode(x, canvas, decoder, width):
    """Optimized stroke decoding with in-place arithmetic."""
    x = x.view(-1, 13)
    # stroke: (Batch, 1, Width, Width)
    stroke = 1.0 - decoder(x[:, :10])
    stroke = stroke.view(-1, 1, width, width)

    # color_stroke: (Batch, 3, Width, Width)
    color_stroke = stroke * x[:, -3:].view(-1, 3, 1, 1)

    # Reshape for the 5-stroke sequence
    # (Batch/5, 5, C, W, W)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)

    res = []
    # Pre-calculating the inverse once for the whole batch
    inv_stroke = 1.0 - stroke

    for i in range(5):
        # Using in-place operations to maximize speed
        canvas = canvas * inv_stroke[:, i] + color_stroke[:, i]
        res.append(canvas)

    return canvas, res


def prepare_output(canvas, to_shape, divide, width, is_divide=False):
    """GPU-based image post-processing."""
    output = canvas.detach()

    if is_divide:
        output = small2large(output, divide, width)
        output = smooth(output)
    else:
        output = output[0]  # (3, W, W)

    # Resize on GPU
    output = output.unsqueeze(0)
    output = F.interpolate(output, size=(to_shape, to_shape), mode='bilinear', align_corners=False)

    # Scale and convert to uint8 (H, W, 3)
    output = (output.squeeze(0) * 255).clamp(0, 255).to(torch.uint8)
    return output.permute(1, 2, 0)


# --- 2. Main Painting Logic ---

def paint(img, output_every, device, actor, renderer):
    max_step = PainterConsts.MAX_STEP
    output_width = img.shape[1]
    canvas_cnt = PainterConsts.DIVIDE * PainterConsts.DIVIDE

    # --- Input Pre-processing (Torch) ---
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img).to(device).float()
    else:
        img = img.to(device).float()

    # Standardize to (1, 3, H, W)
    if img.ndim == 3:
        if img.shape[-1] == 3: img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
    if img.max() > 1.0: img /= 255.0

    # Prepare patches
    patch_img_full = F.interpolate(img, size=(PainterConsts.WIDTH * PainterConsts.DIVIDE,
                                              PainterConsts.WIDTH * PainterConsts.DIVIDE), mode='bilinear')
    patch_img = large2small(patch_img_full, PainterConsts.DIVIDE, PainterConsts.WIDTH,
                            canvas_cnt)  # (25, 3, 128, 128)

    # Prepare global image
    img_low = F.interpolate(img, size=(PainterConsts.WIDTH, PainterConsts.WIDTH), mode='bilinear')  # (1, 3, 128, 128)

    # Prepare Coordinate Grids
    i = torch.linspace(0, 1, PainterConsts.WIDTH, device=device).view(-1, 1).repeat(1, PainterConsts.WIDTH)
    j = torch.linspace(0, 1, PainterConsts.WIDTH, device=device).view(1, -1).repeat(PainterConsts.WIDTH, 1)
    coord = torch.stack([i, j], dim=0).unsqueeze(0)  # (1, 2, 128, 128)

    T = torch.ones([1, 1, PainterConsts.WIDTH, PainterConsts.WIDTH], device=device)
    canvas = torch.zeros([1, 3, PainterConsts.WIDTH, PainterConsts.WIDTH], device=device)

    if PainterConsts.DIVIDE > 1:
        max_step //= 2

    img_idx = 0
    output_canvases = []

    with torch.no_grad():
        # PHASE 1: Regular (Global)
        for i in range(max_step):
            stepnum = T * (i / max_step)
            actor_input = torch.cat([canvas, img_low, stepnum, coord], 1)
            actions = actor(actor_input)
            canvas, res = decode(actions, canvas, renderer, PainterConsts.WIDTH)

            for j in range(5):
                img_idx += 1
                if img_idx in output_every:
                    out = prepare_output(res[j], output_width, PainterConsts.DIVIDE, PainterConsts.WIDTH, False)
                    output_canvases.append(out)

        # PHASE 2: Patched (Local)
        if PainterConsts.DIVIDE > 1:
            # Resize global canvas to match patch resolution
            canvas = F.interpolate(canvas, size=(PainterConsts.WIDTH * PainterConsts.DIVIDE,
                                                 PainterConsts.WIDTH * PainterConsts.DIVIDE), mode='bilinear')
            canvas = large2small(canvas, PainterConsts.DIVIDE, PainterConsts.WIDTH,
                                 canvas_cnt)  # (25, 3, 128, 128)

            # Parallelize: Process all 25 patches as one batch
            coord_p = coord.expand(canvas_cnt, -1, -1, -1)
            T_p = T.expand(canvas_cnt, -1, -1, -1)

            for i in range(max_step):
                stepnum = T_p * (i / max_step)
                actor_input = torch.cat([canvas, patch_img, stepnum, coord_p], 1)
                actions = actor(actor_input)
                canvas, res = decode(actions, canvas, renderer, PainterConsts.WIDTH)

                for j in range(5):
                    img_idx += canvas_cnt
                    if img_idx in output_every:
                        out = prepare_output(res[j], output_width, PainterConsts.DIVIDE, PainterConsts.WIDTH, True)
                        output_canvases.append(out)

    # --- Final Output Formatting ---
    if not output_canvases:
        return torch.empty(0)

    # Stack: (Steps, H, W, 3) -> Convert to (1, Steps, 3, H, W)
    final_out = torch.stack(output_canvases, dim=0).float() / 255.0
    final_out = final_out.permute(0, 3, 1, 2).unsqueeze(0)

    return final_out


def paint_images(x, output_every, device, actor, renderer, add_original=True):
    x_out = []
    # x: (Batch, 3, H, W)
    for i in range(x.shape[0]):
        canvases = paint(x[i], output_every, device, actor, renderer).to(device)
        if add_original:
            orig = x[i:i + 1].unsqueeze(1).to(device) # (1, 1, 3, H, W)
            canvases = torch.cat([canvases, orig], dim=1)
        x_out.append(canvases)
    return torch.cat(x_out, dim=0)


def load_painter(actor_path, renderer_path, device):
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
