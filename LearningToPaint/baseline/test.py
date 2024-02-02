import os
import cv2
import pandas as pd
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F

from DRL.actor import *
from Renderer.stroke_gen import *
from Renderer.model import *


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./model/Paint-run1/actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='image/test.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
parser.add_argument('--output_dir', type=str, help='path to output directory')
parser.add_argument('--output_img_name', type=str, help='output image name')
parser.add_argument('--save_every', type=int, help='save output image in every x steps')
parser.add_argument('--save_strokes', type=bool, help='whether to save generated strokes')
parser.add_argument('--verbose', default=False, type=bool, help='whether to print canvas steps or not')

args = parser.parse_args()

canvas_cnt = args.divide * args.divide
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
img = cv2.imread(args.img, cv2.IMREAD_COLOR)
try:
    origin_shape = (img.shape[1], img.shape[0])
except Exception as e:
    print(f'image was None: {args.img}')
    exit(0)
saving_shape = (300, 300)

coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device) # Coordconv

Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer))

# def decode(x, canvas, verbose=False): # b * (10 + 3)
#     x = x.view(-1, 10 + 3)
#     stroke = 1 - Decoder(x[:, :10])
#
#     stroke = stroke.view(-1, width, width, 1)
#     color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
#     stroke = stroke.permute(0, 3, 1, 2)
#     color_stroke = color_stroke.permute(0, 3, 1, 2)
#     stroke = stroke.view(-1, 5, 1, width, width)
#     color_stroke = color_stroke.view(-1, 5, 3, width, width)
#     if verbose:
#         print(stroke.shape)
#         print(color_stroke.shape)
#     res = []
#     for i in range(5):
#         canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
#         res.append(canvas)
#     return canvas, res
def decode(x, canvas, verbose=False): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])

    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    if verbose:
        print(stroke.shape)
        print(color_stroke.shape)
    res = []
    for j in range(stroke.shape[0]):

        for i in range(5):
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
            res.append(canvas)
    return canvas, res

def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)    
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x

def large2small(x):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, saving_shape)
    output_path = os.path.join(args.output_dir, args.output_img_name + f'_generated{imgid}.png')
    cv2.imwrite(output_path, output)
    # cv2.imwrite('output/generated' + str(imgid) + '.png', output)

actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()
Decoder = Decoder.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)

patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img).to(device).float() / 255.

# os.system('mkdir output')
os.makedirs(args.output_dir, exist_ok=True)

actions_dict = {'image_name': [], 'paint_step': [], 'action_bundle_step': [], 'action_step': [],
                'qbc_x0': [], 'qbc_y0': [], 'qbc_x1': [], 'qbc_y1': [], 'qbc_x2': [], 'qbc_y2': [],
                'thickness_r0': [], 'thickness_t0': [], 'transparency_r1': [], 'transparency_t1': [],
                'R': [], 'G': [], 'B': []}


def save_actions(actions, paint_step):
    actions_bundle = actions.view(-1, 10 + 3)
    action_bundle_n = int(actions_bundle.shape[0]/5)

    for divide_step in range(action_bundle_n):
        bundle_start = divide_step * 5
        bundle_end = bundle_start + 5
        for action_step in range(bundle_start, bundle_end):
            action = actions_bundle[action_step]
            actions_dict['image_name'].append(args.output_img_name)
            actions_dict['paint_step'].append(paint_step)
            actions_dict['action_bundle_step'].append(divide_step+1)
            actions_dict['action_step'].append(action_step+1)
            actions_dict['qbc_x0'].append(round(action[0].item(), 5))
            actions_dict['qbc_y0'].append(round(action[1].item(), 5))
            actions_dict['qbc_x1'].append(round(action[2].item(), 5))
            actions_dict['qbc_y1'].append(round(action[3].item(), 5))
            actions_dict['qbc_x2'].append(round(action[4].item(), 5))
            actions_dict['qbc_y2'].append(round(action[5].item(), 5))
            actions_dict['thickness_r0'].append(round(action[6].item(), 5))
            actions_dict['thickness_t0'].append(round(action[7].item(), 5))
            actions_dict['transparency_r1'].append(round(action[8].item(), 5))
            actions_dict['transparency_t1'].append(round(action[9].item(), 5))
            actions_dict['R'].append(round(action[10].item(), 5))
            actions_dict['G'].append(round(action[11].item(), 5))
            actions_dict['B'].append(round(action[12].item(), 5))


with torch.no_grad():
    if args.divide != 1:
        args.max_step = args.max_step // 2
    for i in range(args.max_step):
        stepnum = T * i / args.max_step
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        canvas, res = decode(actions, canvas)
        if args.verbose:
            print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
        for j in range(5):
            args.imgid += 1
            if (args.imgid + 1) % args.save_every == 0:
                save_img(res[j], args.imgid)



        if args.save_strokes:
            save_actions(actions, i+1)


    if args.divide != 1:
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))    
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        coord = coord.expand(canvas_cnt, 2, width, width)
        T = T.expand(canvas_cnt, 1, width, width)
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
            print(np.array(actions).shape)
            canvas, res = decode(actions, canvas, verbose=True)
            if args.verbose:
                print('divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
            print(f'paint step {41+i} image_id {args.imgid} res shape {np.array(res).shape}')
            for j in range(5):
                args.imgid += args.divide*args.divide
                if (args.imgid + 1) % args.save_every == 0:
                    save_img(res[j], args.imgid, True)
                # args.imgid += 1

            print(f'paint step {41+i} image_id {args.imgid} res shape {np.array(res).shape}')


            if args.save_strokes:
                save_actions(actions, args.max_step + i+1)


if args.save_strokes:
    pd.DataFrame(actions_dict).to_csv(os.path.join(args.output_dir,
                                                   f'strokes_{args.output_img_name}.csv'), index=False)

