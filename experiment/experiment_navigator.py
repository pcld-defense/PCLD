import argparse
from experiment.paint_dataset import main_paint_dataset
from experiment.train_classifier import main_train_classifier
from experiment.attack_pcl import main_attack_pcl
from experiment.train_decisioner import main_train_decisioner
from experiment.attack_pcld import main_attack_pcld


def apply_experiment(args: argparse.Namespace,device: str):
    params = {'args': args,
              'device': device
              }
    if args.experiment_type == 'paint_dataset':
        main_paint_dataset(**params)
    elif args.experiment_type == 'train_classifier':
        main_train_classifier(**params)
    elif args.experiment_type == 'attack_pcl':
        main_attack_pcl(**params)
    elif args.experiment_type == 'attack_pcld':
        main_attack_pcld(**params)
    elif args.experiment_type == 'train_decisioner':
        main_train_decisioner(**params)




