import os
import pickle
import json
import time
import argparse
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import EvaluationRecorder, ResultsSummary, preprocessing_data, makedir, record_results_summary, ParameterDict
# from train_baseline_models import train_LogisticRegression, train_RandomForest, train_XGBoost, train_SVM, train_KNN, train_DecisionTree
from INAFEN import InterpretableAutomatedFeatureEngineering



parameter_dict = {
    'INAFEN': {
        'FC_min_support': [0.2, 0.25, 0.3],
        'FC_min_confidence': [0.8, 0.85, 0.9]
    }
}
parameter_dict = {
    'INAFEN': {
        'FC_min_support': [0.2],
        'FC_min_confidence': [0.85]
    }
}

if __name__ == '__main__':
    desc = "Algorithm Command Line Tool and Library"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_path', type=str, default='./Data')
    parser.add_argument('--save_path', type=str, default='./Results/')
    parser.add_argument('--dataset_name', type=str, default='gmsc.csv')

    parser.add_argument('--DTFT', type=str, default=True)  # tree_based_discretization
    parser.add_argument('--ARFC', type=bool, default=True)
    parser.add_argument('--BMKD', type=bool, default=True)

    max_depth_dict = None
    parser.add_argument('--TBD_max_depth_dict', type=dict, default=max_depth_dict)
    parser.add_argument('--FC_min_support', type=float, default=0.2)
    parser.add_argument('--FC_min_confidence', type=float, default=0.80)
    parser.add_argument('--ST_temperature', type=float, default=1)
    parser.add_argument('--ST_alpha', type=float, default=1)
    args = parser.parse_args()

    # Data reading & preprocessing
    data_df = pd.read_csv(os.path.join(args.data_path,"Raw_data", args.dataset_name))

    data_df, num_categorical_feas, num_numerical_feas = preprocessing_data(data_df, args.dataset_name)
    data_df_columns = data_df.columns[:-1]

    X = data_df.iloc[:, :-1].values
    y = data_df.iloc[:, -1].values

    save_path = os.path.join(args.save_path, args.dataset_name.split('.')[0], 'DTFT[{}]-ARFC[{}]-BMKD[{}]'.format(args.DTFT,
                                                                                                             args.ARFC,
                                                                                                             args.BMKD))
    makedir(save_path=save_path)

    INAFEN_parameter = ParameterDict(parameter_dict)
    parameter_list = INAFEN_parameter.get_parameter_list('INAFEN')

    for parameter_dict in parameter_list:
        valid_evaluation_recorder = EvaluationRecorder()
        test_evaluation_recorder = EvaluationRecorder()
        valid_evaluation_recorder.add_model('INAFEN')
        test_evaluation_recorder.add_model('INAFEN')
        skf_train_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(skf_train_test.split(X, y)):
            print('Using data of fold {}...'.format(i + 1))
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            skf_train_valid = StratifiedKFold(n_splits=4, shuffle=False)
            train_train_index, train_valid_index = list(skf_train_valid.split(X_train, y_train))[0]
            X_train_train, X_train_valid = X_train[train_train_index, :], X_train[train_valid_index, :]
            y_train_train, y_train_valid = y_train[train_train_index], y_train[train_valid_index]

            time_start = time.time()
            teacher_model = XGBClassifier(n_estimators=100, max_depth=3, gamma=1)
            # teacher_model = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_split=3)
            teacher_model.fit(X_train_train, y_train_train)

            INAFEN_model = InterpretableAutomatedFeatureEngineering(DTFT=args.DTFT,
                                                             ARFC=args.ARFC,
                                                             BMKD=args.BMKD,
                                                             number_of_categorical_features=num_categorical_feas,
                                                             number_of_numerical_features=num_numerical_feas,
                                                             TBD_max_depth_dict=args.TBD_max_depth_dict,
                                                             **parameter_dict)
            INAFEN_model.fit(X=pd.DataFrame(X_train_train, columns=data_df_columns),
                      y_true=pd.DataFrame(y_train_train, columns=['classes']),
                      y_soft=teacher_model.predict_proba(X_train_train)[:, 1])
            time_end = time.time()

            valid_evaluation_recorder.add_model_evaluation('INAFEN', INAFEN_model, (INAFEN_model.transform(pd.DataFrame(X_train_valid, columns=data_df_columns)), y_train_valid))
            test_evaluation_recorder.add_model_evaluation('INAFEN', INAFEN_model, (INAFEN_model.transform(pd.DataFrame(X_test, columns=data_df_columns)), y_test))
            valid_evaluation_recorder.evaluation_dict['INAFEN']['run_time'].append(time_end - time_start)
            test_evaluation_recorder.evaluation_dict['INAFEN']['run_time'].append(time_end - time_start)

        file_path = os.path.join(save_path, 'AUC[{:.4f}]-Param[{}].txt'.format(
            np.array(valid_evaluation_recorder.evaluation_dict['INAFEN']['auroc']).mean(),
            parameter_dict).replace(':', '-'))

        with open(file_path, 'w') as f:
            f.write('Model Name: {}\n'.format('INAFEN'))
            f.write('Model Params: {}\n'.format(parameter_dict))
            f.write('Res. of validation set:\n')
            for metric in valid_evaluation_recorder.evaluation_dict['INAFEN'].keys():
                f.write('{}: {:.4f}±{:.4f}\n'.format(metric,
                                                     np.array(valid_evaluation_recorder.evaluation_dict[
                                                                  'INAFEN'][metric]).mean(),
                                                     np.array(valid_evaluation_recorder.evaluation_dict[
                                                                  'INAFEN'][metric]).std()))

            f.write('\n')
            f.write('Res. of test set:\n')
            for metric in test_evaluation_recorder.evaluation_dict['INAFEN'].keys():
                f.write('{}: {:.4f}±{:.4f}\n'.format(metric,
                                                     np.array(test_evaluation_recorder.evaluation_dict[
                                                                  'INAFEN'][metric]).mean(),
                                                     np.array(test_evaluation_recorder.evaluation_dict[
                                                                  'INAFEN'][metric]).std()))




