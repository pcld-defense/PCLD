from util.consts import NUM_OF_HYPHENS, LOCAL_RESULTS_DIR
from util.models import *
from catboost import CatBoostClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def main_evaluate_victim_decisioner(experiment: str, ds_local_path: str, model_decisioner_local_path: str):
    print('-' * NUM_OF_HYPHENS)
    print('Running evaluate victim decisioner experiment...')
    dataset = pd.read_csv(ds_local_path)

    print('round loss to 2 decimal digits')
    dataset['loss_rounded'] = dataset['loss'].round(2)

    print('add image_name_short, paint_step and epsilon_number columns')
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

    def eval_catboost(model, train, val, test, features):
        x_train = train[features]
        x_val = val[features]
        x_test = test[features]

        # predictions on train
        pred_probs = model.predict_proba(x_train)
        pred_label = model.predict(x_train)
        train['decisioner_pred_prob_cat'] = pred_probs[:, 0]
        train['decisioner_pred_prob_dog'] = pred_probs[:, 1]
        train['decisioner_pred_prob_squirrel'] = pred_probs[:, 2]
        train['decisioner_pred_label'] = pred_label

        # predictions on validation
        pred_probs = model.predict_proba(x_val)
        pred_label = model.predict(x_val)
        val['decisioner_pred_prob_cat'] = pred_probs[:, 0]
        val['decisioner_pred_prob_dog'] = pred_probs[:, 1]
        val['decisioner_pred_prob_squirrel'] = pred_probs[:, 2]
        val['decisioner_pred_label'] = pred_label

        # predictions on test
        pred_probs = model.predict_proba(x_test)
        pred_label = model.predict(x_test)
        test['decisioner_pred_prob_cat'] = pred_probs[:, 0]
        test['decisioner_pred_prob_dog'] = pred_probs[:, 1]
        test['decisioner_pred_prob_squirrel'] = pred_probs[:, 2]
        test['decisioner_pred_label'] = pred_label

        results = pd.concat([train, val, test], axis=0, ignore_index=True)

        return results

    # download pre-trained catboost from s3
    print('load pre-trained catboost')
    model = CatBoostClassifier()
    model.load_model(model_decisioner_local_path)
    print(f'pretrained decisioner model load from: {model_decisioner_local_path}')

    print('evaluate decisioner')
    train = dataset_pivot[dataset_pivot.ds_name == 'train']
    val = dataset_pivot[dataset_pivot.ds_name == 'validation']
    test = dataset_pivot[dataset_pivot.ds_name == 'test']
    dataset_pivot = eval_catboost(model, train, val, test, features)

    print('save results')
    results_local_dir = os.path.join(LOCAL_RESULTS_DIR, experiment)
    results_local_path = os.path.join(results_local_dir, 'results_decisioner.csv')
    os.makedirs(results_local_dir, exist_ok=True)
    dataset_pivot.to_csv(results_local_path, index=False)

    print('Finished running evaluate victim decisioner experiment!')
