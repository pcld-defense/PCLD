import argparse
from experiment.train_victim_clf import main_train_victim_clf
from experiment.train_victim_decisioner import main_train_victim_decisioner
from experiment.train_surrogate_for_victim_clf import main_train_surrogate_for_victim_clf
from experiment.train_surrogate_for_victim_clf_undefended import main_train_surrogate_for_victim_clf_undefended
from experiment.train_surrogate_for_victim_decisioner import main_train_surrogate_for_victim_decisioner
from experiment.train_adv_madry_victim_clf import main_train_adv_madry_victim_clf
from experiment.attack_surrogate_for_victim_clf import main_attack_surrogate_for_victim_clf
from experiment.attack_surrogate_for_victim_decisioner import main_attack_surrogate_for_victim_decisioner
from experiment.defend_drawing_agent import main_defend_drawing_agent
from experiment.evaluate_victim_clf import main_evaluate_victim_clf
from experiment.evaluate_victim_decisioner import main_evaluate_victim_decisioner


def check_hyper_params(params):
    for n, p in params.items():
        assert p, f"{n} is missing"


def apply_experiment(args: argparse.Namespace, experiment: str,  device: str):

    if args.experiment_type == 'train_victim_clf':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'batch_size': args.batch_size,
                  'lr': args.lr,
                  'n_epochs': args.n_epochs
                  }
        check_hyper_params(params)
        main_train_victim_clf(**params)

    elif args.experiment_type == 'train_victim_decisioner':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path
                  }
        check_hyper_params(params)
        main_train_victim_decisioner(**params)

    elif args.experiment_type == 'train_surrogate_for_victim_clf':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'batch_size': args.batch_size,
                  'lr': args.lr,
                  'n_epochs': args.n_epochs,
                  'net_victim_local_dir': args.net_victim_local_dir,
                  'ds_victim_local_path': args.ds_victim_local_path
                  }
        check_hyper_params(params)
        main_train_surrogate_for_victim_clf(**params)

    elif args.experiment_type == 'train_surrogate_for_victim_clf_undefended':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'batch_size': args.batch_size,
                  'lr': args.lr,
                  'n_epochs': args.n_epochs,
                  'net_victim_local_dir': args.net_victim_local_dir
                  }
        check_hyper_params(params)
        main_train_surrogate_for_victim_clf_undefended(**params)

    elif args.experiment_type == 'train_surrogate_for_victim_decisioner':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'batch_size': args.batch_size,
                  'lr': args.lr,
                  'n_epochs': args.n_epochs,
                  'ds_decisioner_local_path': args.ds_decisioner_local_path
                  }
        check_hyper_params(params)
        main_train_surrogate_for_victim_decisioner(**params)

    elif args.experiment_type == 'train_adv_madry_victim_clf':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'batch_size': args.batch_size,
                  'lr': args.lr,
                  'n_epochs': args.n_epochs,
                  'epsilons': args.epsilons
                  }
        check_hyper_params(params)
        main_train_adv_madry_victim_clf(**params)

    elif args.experiment_type == 'attack_surrogate_for_victim_clf':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'batch_size': args.batch_size,
                  'epsilons': args.epsilons,
                  'net_surrogate_local_dir': args.net_surrogate_local_dir,
                  'net_victim_local_dir': args.net_victim_local_dir,
                  'attack': args.attack
                  }
        check_hyper_params(params)
        main_attack_surrogate_for_victim_clf(**params)

    elif args.experiment_type == 'attack_surrogate_for_victim_decisioner':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'batch_size': args.batch_size,
                  'epsilons': args.epsilons,
                  'net_surrogate_local_dir': args.net_surrogate_local_dir,
                  'attack': args.attack
                  }
        check_hyper_params(params)
        main_attack_surrogate_for_victim_decisioner(**params)

    elif args.experiment_type == 'defend_drawing_agent':
        params = {'ds_local_path': args.ds_local_path,
                  'save_every': args.save_every
                  }
        check_hyper_params(params)
        main_defend_drawing_agent(**params)

    elif args.experiment_type == 'evaluate_victim_clf':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'device': device,
                  'net_victim_local_dir': args.net_victim_local_dir,
                  'batch_size': args.batch_size
                  }
        check_hyper_params(params)
        main_evaluate_victim_clf(**params)

    elif args.experiment_type == 'evaluate_victim_decisioner':
        params = {'experiment': experiment,
                  'ds_local_path': args.ds_local_path,
                  'model_decisioner_local_path': args.model_decisioner_local_path
                  }
        check_hyper_params(params)
        main_evaluate_victim_decisioner(experiment=experiment,
                                        ds_local_path=args.ds_local_path,
                                        model_decisioner_local_path=args.model_decisioner_local_path)



