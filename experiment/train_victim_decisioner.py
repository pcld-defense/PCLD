from util.models import *
from util.consts import NUM_OF_HYPHENS, LOCAL_RESULTS_DIR, LOCAL_MODELS_DIR
from catboost import CatBoostClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def balance_epsilon(df):
    eps_0_data = df[df['dir'] == 'eps_0']
    to_concat = [eps_0_data for _ in range(df['dir'].nunique())]
    balanced_df = pd.concat([df] + to_concat, axis=0, ignore_index=True)
    return balanced_df


def main_train_victim_decisioner(experiment: str, ds_local_path: str):
    print('-' * NUM_OF_HYPHENS)
    print('Running train victim decisioner experiment...')

    print('read the dataset')
    dataset = pd.read_csv(ds_local_path)

    # round loss to 2 decimal digits
    dataset['loss_rounded'] = dataset['loss'].round(2)

    # add image_name_short, paint_step and epsilon_number columns
    splitted = dataset['image_name'].str.split('_generated', expand=True)
    splitted.columns = ['image_name_short', 'paint_step']
    splitted.fillna('9999', inplace=True)
    splitted['paint_step'] = splitted['paint_step'].astype(np.int64)
    dataset['image_name_short'] = splitted.iloc[:, 0]
    dataset['paint_step'] = splitted.iloc[:, 1]
    dataset.sort_values(by=['ds_name', 'dir', 'image_name_short', 'paint_step'], inplace=True)

    def pivot_data(data):
        x = data.pivot(index=['experiment', 'dir', 'ds_name', 'image_name_short', 'real_label', 'real_label_name'],
                       columns=['paint_step'],
                       values=['pred_prob_cat', 'pred_prob_dog', 'pred_prob_squirrel']).reset_index()
        x.columns = [c[0] + ('_'+str(c[1]) if len(str(c[1])) > 0 else '') for c in x.columns]
        return x

    # flattening predictions by paint steps
    dataset_pivot = pivot_data(dataset)
    features = [c for c in dataset_pivot.columns if 'pred_prob_' in c]
    label = 'real_label'

    def train_eval_catboost(train_full, train, val, test, features, label):

        train_full_epsilon = train_full['dir'].str.split('_', expand=True).iloc[:, 1].astype(np.int64)
        train_epsilon = train['dir'].str.split('_', expand=True).iloc[:, 1].astype(np.int64)
        sample_weight_train_full = [(40 if e == 0 else 10 if e <= 15 else 5) for e in train_full_epsilon]
        sample_weight_train = [(40 if e == 0 else 10 if e <= 15 else 5) for e in train_epsilon]

        x_train_full = train_full[features]
        x_train = train[features]
        x_val = val[features]
        x_test = test[features]
        y_train_full = train_full[label]
        y_train = train[label]
        y_val = val[label]
        y_test = test[label]

        # =================> train-validation and test <================= #
        print('Train a temp model for searching the number of trees')
        cb_train_best_model = None
        best_score = 0
        for max_depth in [2, 3, 4, 5, 6, 7, 9, 11, 13, 15]:
            cb_train = CatBoostClassifier(random_state=42, n_estimators=7000, max_depth=max_depth,
                                          od_type="Iter", od_wait=50)
            cb_train.fit(x_train, y_train, verbose=False, eval_set=(x_val, y_val), sample_weight=sample_weight_train)
            model_score = cb_train.score(x_val, y_val)
            print(f'depth {max_depth} score: {model_score}')
            if model_score > best_score:
                cb_train_best_model = cb_train
                best_score = model_score
                print(f'set model with depth {max_depth} as the best model')

        print('Train the final model with the best iterations found')
        cb_train_full = CatBoostClassifier(random_state=42, n_estimators=cb_train_best_model.get_best_iteration()+1)
        cb_train_full.fit(x_train_full, y_train_full, verbose=False, sample_weight=sample_weight_train_full)

        print('Run predictions on train')
        pred_probs = cb_train_full.predict_proba(x_train)
        pred_label = cb_train_full.predict(x_train)
        train['decisioner_pred_prob_cat'] = pred_probs[:, 0]
        train['decisioner_pred_prob_dog'] = pred_probs[:, 1]
        train['decisioner_pred_prob_squirrel'] = pred_probs[:, 2]
        train['decisioner_pred_label'] = pred_label

        print('Run predictions on validation')
        pred_probs = cb_train_full.predict_proba(x_val)
        pred_label = cb_train_full.predict(x_val)
        val['decisioner_pred_prob_cat'] = pred_probs[:, 0]
        val['decisioner_pred_prob_dog'] = pred_probs[:, 1]
        val['decisioner_pred_prob_squirrel'] = pred_probs[:, 2]
        val['decisioner_pred_label'] = pred_label

        print('Run predictions on test')
        pred_probs = cb_train_full.predict_proba(x_test)
        pred_label = cb_train_full.predict(x_test)
        test['decisioner_pred_prob_cat'] = pred_probs[:, 0]
        test['decisioner_pred_prob_dog'] = pred_probs[:, 1]
        test['decisioner_pred_prob_squirrel'] = pred_probs[:, 2]
        test['decisioner_pred_label'] = pred_label

        results = pd.concat([train, val, test], axis=0, ignore_index=True)

        return results, cb_train_full

    # train catboost and generate predictions
    train_full = dataset_pivot[dataset_pivot.ds_name.isin(['train', 'validation'])]
    train = dataset_pivot[dataset_pivot.ds_name == 'train']
    val = dataset_pivot[dataset_pivot.ds_name == 'validation']
    test = dataset_pivot[dataset_pivot.ds_name == 'test']


    dataset_pivot, cb_train_full = train_eval_catboost(train_full, train, val, test, features, label)

    # Save results & model
    # Results df
    results_local_dir = os.path.join(LOCAL_RESULTS_DIR, experiment)
    model_local_dir = os.path.join(LOCAL_MODELS_DIR, experiment)
    results_local_path = os.path.join(results_local_dir, 'results_decisioner.csv')
    model_local_path = os.path.join(model_local_dir, 'model_decisioner.csv')
    os.makedirs(results_local_dir, exist_ok=True)
    os.makedirs(model_local_dir, exist_ok=True)
    results_df = pd.DataFrame()
    results_deep_df = pd.DataFrame()

    print('save results & model')
    dataset_pivot.to_csv(results_local_path, index=False)
    cb_train_full.save_model(model_local_path)

    print('Finished running train victim decisioner experiment!')
