import os.path

import cv2
import numpy as np
import torch
from scipy.ndimage import convolve

from util.consts import NUM_OF_HYPHENS, RESOURCES_MODELS_DIR


def large2small(x, divide, width, canvas_cnt):
    # (width * divide, width * divide, 3) -> (divide, divide, width, width, 3)
    x = x.reshape(divide, width, divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    # (divide, divide, width, width, 3) -> (canvas_cnt, width, width, 3)
    x = x.reshape(canvas_cnt, width, width, 3)
    return x


def small2large(x, divide, width):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(divide, divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(divide * width, divide * width, -1)
    return x


def decode(x, canvas, decoder, width):  # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - decoder(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res



# def smooth(img, divide, width):
#     def smooth_pix(img, tx, ty):
#         if tx == divide * width - 1 or ty == divide * width - 1 or tx == 0 or ty == 0:
#             return img
#         img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
#         return img

#     for p in range(divide):
#         for q in range(divide):
#             x = p * width
#             y = q * width
#             for k in range(width):
#                 img = smooth_pix(img, x + k, y + width - 1)
#                 if q != divide - 1:
#                     img = smooth_pix(img, x + k, y + width)
#             for k in range(width):
#                 img = smooth_pix(img, x + width - 1, y + k)
#                 if p != divide - 1:
#                     img = smooth_pix(img, x + width, y + k)
#     return img


def smooth(img, divide, width):
    # Ensure the kernel is applied independently to each color channel
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9.0

    # Convolve each channel separately if img is an RGB image
    if img.ndim == 3 and img.shape[2] == 3:
        smoothed_img = np.zeros_like(img)
        for i in range(3):  # Process each channel
            smoothed_img[:, :, i] = convolve(img[:, :, i], kernel, mode='nearest')
    else:
        smoothed_img = convolve(img, kernel, mode='nearest')

    # Restore boundary pixels to original (preserving edges explicitly)
    # Top and bottom rows
    smoothed_img[0, :] = img[0, :]
    smoothed_img[-1, :] = img[-1, :]
    # Left and right columns
    smoothed_img[:, 0] = img[:, 0]
    smoothed_img[:, -1] = img[:, -1]

    return smoothed_img



def prepare_output(canvas, to_shape,
                   divide, device, is_divide=False, width=300):
    output = canvas.detach().cpu().numpy()  # (divide * divide, 3, width, width)
    output = np.transpose(output, (0, 2, 3, 1))
    if is_divide:
        output = small2large(output, divide, width)
        output = smooth(output, divide, width)
    else:
        output = output[0]

    output = output * 255
    output = output.astype('uint8')
    output = cv2.resize(output, (to_shape, to_shape))
    output = torch.tensor(output).to(device)
    return output



def paint(img, output_every, device, actor, renderer):
    output_width = 300  # imagenet like
    max_step = 80
    width = 128
    divide = 5
    canvas_cnt = divide * divide
    verbose = False
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device) # -> (1, 1, 128, 128)

    # Coordconv
    i = torch.arange(width).view(-1, 1).float() / (width - 1)  # Column vector
    j = torch.arange(width).view(1, -1).float() / (width - 1)  # Row vector
    coord = torch.stack([i.repeat(1, width), j.repeat(width, 1)], dim=0)
    coord = coord.unsqueeze(0)
    coord = coord.to(device) # -> (1, 2, 128, 128)

    # canvas
    canvas = torch.zeros([1, 3, width, width]).to(device) # -> (1, 3, 128, 128)

    if isinstance(img, torch.Tensor):
        img = img.to('cpu').detach().numpy()  # Convert to numpy array
    if img.ndim == 3 and img.shape[2] != 3:  # Check if image has 3 channels
        img = np.transpose(img, (1, 2, 0))  # Reorder dimensions if needed

    # prepare the patched image
    patch_img = cv2.resize(img, (width * divide, width * divide)) # -> (640, 640, 3)
    patch_img = large2small(patch_img, divide, width, canvas_cnt) # -> (25, 128, 128, 3)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2)) # -> (25, 3, 128, 128)
    patch_img = torch.tensor(patch_img).to(device).float()

    # prepare the image
    img = cv2.resize(img, (width, width)) # -> (128, 128, 3)
    img = img.reshape(1, width, width, 3) # -> (1, 128, 128, 3)
    img = np.transpose(img, (0, 3, 1, 2)) # -> (1, 3, 128, 128)
    img = torch.tensor(img).to(device).float()

    # divide the painting to two phases (regular phase & patched phase)
    if divide > 1:
        max_step //= 2

    img_idx = 0
    output_canvases = []
    with torch.no_grad():
        # regular phase
        for i in range(max_step):
            stepnum = T * i / max_step
            actor_input = torch.cat([canvas, img, stepnum, coord], 1)
            actions = actor(actor_input)
            canvas, res = decode(actions, canvas, renderer, width)
            if verbose:
                print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
            for j in range(5):
                img_idx += 1
                if img_idx in output_every:
                    output_canvas = prepare_output(res[j], output_width, divide, device,
                                                   is_divide=False, width=width)
                    output_canvases.append(output_canvas)

        # patched phase
        if divide > 1:
            canvas = canvas[0].detach().cpu().numpy()
            canvas = np.transpose(canvas, (1, 2, 0))
            canvas = cv2.resize(canvas, (width * divide, width * divide))
            canvas = large2small(canvas, divide, width, canvas_cnt)
            canvas = np.transpose(canvas, (0, 3, 1, 2))
            canvas = torch.tensor(canvas).to(device).float()
            coord = coord.expand(canvas_cnt, 2, width, width)
            T = T.expand(canvas_cnt, 1, width, width)
            for i in range(max_step):
                stepnum = T * i / max_step
                actor_input = torch.cat([canvas, patch_img, stepnum, coord], 1)
                actions = actor(actor_input)
                canvas, res = decode(actions, canvas, renderer, width)
                if verbose:
                    print('divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
                for j in range(len(res)):
                    img_idx += divide ** 2
                    if img_idx in output_every:
                        output_canvas = prepare_output(res[j], output_width, divide, device,
                                                       is_divide=True, width=width)
                        output_canvases.append(output_canvas)
    output_canvases = torch.stack(output_canvases, dim=0).to(device)
    output_canvases = output_canvases.type(torch.float32)
    output_canvases = output_canvases.view((1, ) + tuple(output_canvases.shape))
    output_canvases /= 255.
    output_canvases = output_canvases.permute(0, 1, 4, 2, 3)
    # output_canvases = output_canvases.view(-1, 3, 300, 300)
    return output_canvases


# def paint_images(x, output_every, device, actor, renderer):
#     x_out = []
#     for i in range(x.shape[0]):
#         canvases = paint(x[i], output_every, device, actor, renderer)
#         # print(canvases.shape, x[i:i+1].unsqueeze(1).shape)
#         # add the original image (t=∞) as well
#         canvases = torch.cat([canvases, x[i:i+1].unsqueeze(1)], dim=1)
#         x_out.append(canvases)
#     x_out = torch.cat(x_out, dim=0)
#     return x_out

def paint_images(x, output_every, device, actor, renderer, add_original=True):
    x_out = []
    for i in range(x.shape[0]):
        canvases = paint(x[i], output_every, device, actor, renderer)
        # print(canvases.shape, x[i:i+1].unsqueeze(1).shape)
        if add_original:
            # add the original image (t=∞) as well
            canvases = torch.cat([canvases, x[i:i+1].unsqueeze(1)], dim=1)
        x_out.append(canvases)
    x_out = torch.cat(x_out, dim=0)
    return x_out



from model.painter import ActorResNet, RendererFCN
def load_painter(device):
    print('-' * NUM_OF_HYPHENS)
    print('Loading painter...')
    actor_path = os.path.join(RESOURCES_MODELS_DIR, 'painter_actor/actor.pkl')
    renderer_path = os.path.join(RESOURCES_MODELS_DIR, 'painter_renderer/renderer.pkl')
    if not os.path.exists(actor_path) or not os.path.exists(renderer_path):
        raise Exception(f'Missing actor or renderer: \n{actor_path}\n{renderer_path}')
    actor = ActorResNet(9, 18, 65) # 65 = 5 (action_bundle) * 13 (stroke parameters)
    actor.load_state_dict(torch.load(actor_path))
    renderer = RendererFCN()
    renderer.load_state_dict(torch.load(renderer_path))

    actor = actor.to(device).eval()
    renderer = renderer.to(device).eval()
    print('Finished loading painter!')

    return actor, renderer

