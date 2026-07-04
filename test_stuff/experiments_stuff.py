import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from robustbench.data import load_imagenet

from pcld.models.decisioner import DecisionerFC
from pcld.painter.painter_surrogate import load_painter_surrogate
from pcld.painter.painter import ActorResNet, RendererFCN
from pcld.painter.painter_surrogate import IdentitySurrogate_, PainterSurrogate
from pcld.painter.painter_utils import paint_images
from pcld.attacks.pcld_bpda import PCLD, BPDAPainter
from pcld.utils.consts import IMAGENET_2012_LABELS, IMAGENET_7_LABELS_NEW
from pcld.models.train_utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

x_test, y_test = load_imagenet(n_examples=500, data_dir='../data/')
samples_x, samples_y = [], []

print(y_test[0].item(), x_test[0].shape)
for x, y, in zip(x_test, y_test):
    if y.item() in IMAGENET_7_LABELS_NEW.keys():
        print("added")
        img_resized = F.interpolate(x.unsqueeze(0), size=(300, 300), mode='bilinear', align_corners=False).squeeze(0)
        new_x = img_resized.permute(1, 2, 0)
        samples_x.append(new_x)
        samples_y.append(y)

num_classes = len(IMAGENET_2012_LABELS.values())
names_classes = list(IMAGENET_2012_LABELS.values())
num_paint_steps = 16
output_every = [50, 100, 200, 300, 400, 500, 600, 700, 950, 1200, 1700, 2200, 3200, 4200, 5200]


# classifier
clf_images_paints_path = f'../resources/models/train_victim_clf_bp/model.pth'
clf_images_paints = models.resnet18()
clf_images_paints.fc = nn.Linear(clf_images_paints.fc.in_features, num_classes) # 7 classes
clf_images_paints = load_model(clf_images_paints, clf_images_paints_path, device)
clf_images_paints = clf_images_paints.to(device)
clf_images_paints.eval()

# decisioner
load_model_path = f'../resources/models/train_decisioner_fc_fgsm/model.pth'
decisioner = DecisionerFC(num_classes, num_paint_steps).to(device)
decisioner = load_model(decisioner, load_model_path, device)
decisioner.eval()

# painter
actor_path = f'../resources/models/painter_actor/actor.pkl'
renderer_path = f'../resources/models/painter_renderer/renderer.pkl'
actor = ActorResNet(9, 18, 65) # 65 = 5 (action_bundle) * 13 (stroke parameters)
renderer = RendererFCN()
actor.load_state_dict(torch.load(actor_path))
renderer.load_state_dict(torch.load(renderer_path))
actor = actor.to(device).eval()
renderer = renderer.to(device).eval()

# surrogate
surrogates_folder = f'./resources/models/train_surrogate_painter'
surrogate_list = load_painter_surrogate(surrogates_folder, device, output_every)
# add the image itself (t=∞)
surrogate_list.append(IdentitySurrogate_().to(device))
[s.eval() for s in surrogate_list]
painter_surrogate = PainterSurrogate(surrogate_list)
bpda_painter = BPDAPainter(paint_images, painter_surrogate, output_every, device, actor, renderer)
bpda_painter = bpda_painter.to(device)
bpda_painter.eval()

pcld_adversary = PCLD(bpda_painter, clf_images_paints, decisioner, num_paint_steps, 'fc').to(device).eval()

for img, y in zip(samples_x, samples_y):
    img = torch.tensor(np.expand_dims(img, axis=0)).to(device)
    img = img.permute(0, 3, 1, 2)
    output = pcld_adversary(img)
    probs = torch.softmax(output, dim=1).cpu().argmax()
    print(probs, y)
#
# # Evaluate the Linf robustness of the model using AutoAttack
# clean_acc, robust_acc = benchmark(pcld_adversary,
#                                   dataset='',
#                                   threat_model='Linf',
#                                   device=device,
#                                   eps=0.8,
#                                   to_disk=True)
#
# print(clean_acc, robust_acc)
