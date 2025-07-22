import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss

import os
import itertools


class ResultsSummary:
    def __init__(self, dataset_name, model_list, save_path, discretization_method, feature_generation, soft_training):
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.model_list = model_list

        self.discretization_method = discretization_method
        self.feature_generation = feature_generation
        self.soft_training = soft_training

        self.results = {}
        for model in model_list:
            self.results[model] = {'auroc': [],
                                   'auprc': [],
                                   'brier': [],
                                   'f1': [],
                                   'accuracy': [],
                                   'precision': [],
                                   'recall': []}

    def return_results_summary(self):
        summary_dict = {}
        for model in self.model_list:
            summary_dict[model] = {}
            for metric in ['auroc', 'auprc', 'brier', 'f1', 'accuracy', 'precision', 'recall']:
                summary_dict[model][metric] = '{:.4f} +- {:.4f}'.format(np.array(self.results[model][metric]).mean(),
                                                                       np.array(self.results[model][metric]).std())
        return summary_dict


class EvaluationRecorder:
    def __init__(self):
        self.evaluation_dict = {}

    def add_model(self, model_name):
        if isinstance(model_name, str):
            self.evaluation_dict[model_name] = {'accuracy': [],
                                                'precision': [],
                                                'recall': [],
                                                'f1': [],
                                                'auroc': [],
                                                'auprc': [],
                                                'brier': [],
                                                'run_time': []}
        elif isinstance(model_name, list):
            for name_ in model_name:
                self.evaluation_dict[name_] = {'accuracy': [],
                                                'precision': [],
                                                'recall': [],
                                                'f1': [],
                                                'auroc': [],
                                                'auprc': [],
                                                'brier': [],
                                                'run_time': []
                                               }

    def add_model_evaluation(self, model_name, model, eval_dataset):
            self.evaluation_dict[model_name]['accuracy'].append(
                accuracy_score(eval_dataset[1], model.predict(eval_dataset[0])))
            self.evaluation_dict[model_name]['precision'].append(
                precision_score(eval_dataset[1], model.predict(eval_dataset[0])))
            self.evaluation_dict[model_name]['recall'].append(
                recall_score(eval_dataset[1], model.predict(eval_dataset[0])))
            self.evaluation_dict[model_name]['f1'].append(
                f1_score(eval_dataset[1], model.predict(eval_dataset[0])))
            self.evaluation_dict[model_name]['auroc'].append(
                roc_auc_score(eval_dataset[1], model.predict_proba(eval_dataset[0])[:, 1]))
            self.evaluation_dict[model_name]['auprc'].append(
                average_precision_score(eval_dataset[1], model.predict_proba(eval_dataset[0])[:, 1]))
            self.evaluation_dict[model_name]['brier'].append(
                brier_score_loss(eval_dataset[1], model.predict_proba(eval_dataset[0])[:, 1]))


    def add_evaluation_score(self, model_name, metrics, value):
        self.evaluation_dict[model_name][metrics].append(value)


class TrainedModel:
    def __init__(self, model_name, trained_model):
        self.model_name = model_name
        self.trained_model = trained_model
        self.vallidation_metrics = {}

    def record_metrics(self, metrics, value):
        self.vallidation_metrics[metrics] = value

    def get_soft_labels(self, X):
        return self.trained_model.predict_proba(X)[:, 1]


def preprocessing_data(data_df, dataset_name):
    if dataset_name == 'dataset_31_credit-g.csv':
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = list(np.array(range(df_feas.shape[1]))[df_feas.dtypes=='object'])
        numerical_cols = list(np.array(range(df_feas.shape[1]))[df_feas.dtypes == 'int64'])
        for col in ['installment_commitment', 'residence_since']:
            cat_cols.append(list(df_feas.columns).index(col))
            numerical_cols.remove(list(df_feas.columns).index(col))

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset_53_heart-statlog.csv':
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [1, 2, 5, 6, 8, 10, 11, 12]
        numerical_cols = [0, 3, 4, 7, 9]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset_37_diabetes.csv':
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = []
        numerical_cols = [0, 1, 2, 3, 4, 5, 6, 7]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset_heart-failure.csv':
        # 结果挺好
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [1, 3, 5, 9, 10, ]
        numerical_cols = [0, 2, 4, 6, 7, 8, 11]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset_29_credit-a.csv':
        data_df.columns = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity',
                           'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen',
                           'ZipCode', 'Income', 'ApprovalStatus']
        data_df['Age'] = data_df['Age'].replace('?', np.nan)
        data_df['Age'] = data_df['Age'].astype(float)
        data_df['Age'].fillna(data_df['Age'].mode()[0], inplace=True)

        data_df['Gender'] = data_df['Gender'].replace('?', np.nan)
        data_df['Married'] = data_df['Married'].replace('?', np.nan)
        data_df['BankCustomer'] = data_df['BankCustomer'].replace('?', np.nan)
        data_df['EducationLevel'] = data_df['EducationLevel'].replace('?', np.nan)
        data_df['Ethnicity'] = data_df['Ethnicity'].replace('?', np.nan)
        data_df['ZipCode'] = data_df['ZipCode'].replace('?', np.nan)

        categorical = [var for var in data_df.columns if data_df[var].dtype == 'O']
        for col in categorical:
            data_df[col].fillna(data_df[col].mode()[0], inplace=True)

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [0, 3, 4, 5, 6, 8, 9, 11, 12]
        numerical_cols = [1, 2, 7, 10, 13, 14]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target
        df_preprocessed.drop(['ZipCode'], axis=1, inplace=True)

    elif dataset_name == 'dataset-wdbc.csv':
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = []
        numerical_cols = list(range(30))

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset_55_hepatitis.csv':

        missing_cols = list(data_df.columns[(data_df=='?').sum()>0])
        for missing_col in missing_cols:
            data_df[missing_col] = data_df[missing_col].replace('?', np.nan)

        cat_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
        numerical_cols = [0, 13, 14, 15, 16, 17]

        for numerical_col in numerical_cols:
            data_df.iloc[:, numerical_col][data_df.iloc[:, numerical_col].notnull()] = data_df.iloc[:, numerical_col][data_df.iloc[:, numerical_col].notnull()].astype('float')

        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        data_df.iloc[:, cat_cols] = cat_imputer.fit_transform(data_df.iloc[:, cat_cols])
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset-risk-factors-cervical.csv':
        missing_cols = list(data_df.columns[(data_df == '?').sum() > 0])
        for missing_col in missing_cols:
            data_df[missing_col] = data_df[missing_col].replace('?', np.nan)

        cat_cols = [4, 7, 10, 11, 12, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34]
        numerical_cols = [0, 1, 2, 3, 5, 6, 8, 9, 25, 26, 27]

        for numerical_col in numerical_cols:
            data_df.iloc[:, numerical_col][data_df.iloc[:, numerical_col].notnull()] = data_df.iloc[:, numerical_col][
                data_df.iloc[:, numerical_col].notnull()].astype('float')

        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        data_df.iloc[:, cat_cols] = cat_imputer.fit_transform(data_df.iloc[:, cat_cols])
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset-Indian-Liver-Patient-Dataset.csv':
        data_df.loc[data_df.AG.isnull(), 'AG'] = data_df.AG.mean()

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [1]
        numerical_cols = [0, 2, 3, 4, 5, 6, 7, 8, 9]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'dataset-chronic-kidney-disease.csv':
        data_df.drop(['id'], axis=1, inplace=True)
        data_df.classification[data_df.classification=="'ckd\\t'"] = 'ckd'

        cat_cols = [5, 6, 7, 8, 18, 19, 20, 21, 22, 23]
        numerical_cols = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17]

        missing_cols = list(data_df.columns[(data_df == '?').sum() > 0])
        for missing_col in missing_cols:
            data_df[missing_col] = data_df[missing_col].replace('?', np.nan)
            data_df[missing_col] = data_df[missing_col].replace("'\\t?'", np.nan)
            data_df[missing_col] = data_df[missing_col].replace("'\\t43'", np.nan)
            data_df[missing_col] = data_df[missing_col].replace("'\\t6200'", np.nan)
            data_df[missing_col] = data_df[missing_col].replace("'\\t8400'", np.nan)

        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        # print(data_df)
        # exit()

        data_df.iloc[:, cat_cols] = cat_imputer.fit_transform(data_df.iloc[:, cat_cols])
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'diabetes_data_upload.csv':
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = list(range(1, 16, 1))
        numerical_cols = [0]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'South-German-Credit-Prediction.csv':
        data_df.drop(['ID'], axis=1, inplace=True)

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [2, 3, 5, 8, 9, 11, 13, 14, 16, 17, 18, 19]
        numerical_cols = [0, 1, 4, 6, 7, 10, 12, 15, ]


        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'HCC-survival.csv':

        missing_cols = list(data_df.columns[(data_df == '?').sum() > 0])
        for missing_col in missing_cols:
            data_df[missing_col] = data_df[missing_col].replace('?', np.nan)

        cat_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 43, ]
        numerical_cols = [23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48]

        for numerical_col in numerical_cols:
            data_df.iloc[:, numerical_col][data_df.iloc[:, numerical_col].notnull()] = data_df.iloc[:, numerical_col][
                data_df.iloc[:, numerical_col].notnull()].astype('float')

        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        data_df.iloc[:, cat_cols] = cat_imputer.fit_transform(data_df.iloc[:, cat_cols])
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'SEER-Breast-Cancer-Dataset.csv':


        print(data_df.columns)
        exit()
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11]
        numerical_cols = [0, 9, 12, 13, 14]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'HelocData.csv':

        data_df = data_df.loc[(data_df == -9).sum(1) < 20, :].reset_index().iloc[:, 1:]
        data_df = data_df.replace([-7, -8, -9], np.nan)

        mean_imputing_cols = ['x1', 'x2', 'x9', 'x18', 'x19', 'x23']
        mode_imputing_cols = ['x15', 'x20', 'x21', 'x22']

        mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_df[mean_imputing_cols] = mean_imputer.fit_transform(data_df[mean_imputing_cols])

        mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        data_df[mode_imputing_cols] = mode_imputer.fit_transform((data_df[mode_imputing_cols]))

        df_feas = data_df.iloc[:, 1:]
        df_target = data_df.iloc[:, 0]

        cat_cols = []
        numerical_cols = list(range(23))

        num_categorical_feas = 0
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target


    elif dataset_name == 'Pima_Indians_diabetes.csv':

      df_feas = data_df.iloc[:, :-1]
      df_target = data_df.iloc[:, -1]

      numerical_cols = df_feas.columns.tolist()  # all feature columns are numerical
      num_categorical_feas = 0
      num_numerical_feas = len(numerical_cols)


      # Scale numerical features (optional but useful)
      scaler = StandardScaler()
      df_feas_scaled = pd.DataFrame(scaler.fit_transform(df_feas), columns=df_feas.columns)

      # Encode target if necessary (already 0/1, so this is optional)
      label_encoder = LabelEncoder()
      df_target_encoded = pd.DataFrame(label_encoder.fit_transform(df_target), columns=['classes'])

      # Combine into one preprocessed dataframe
      df_preprocessed = pd.concat([df_feas_scaled, df_target_encoded], axis=1)


    elif dataset_name == 'heart.csv':

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [1, 2, 5, 6, 8, 10, 11, 12]
        numerical_cols = [0, 3, 4, 7, 9]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'Stomach_cohort_sampled.csv':
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        numerical_cols = [9, 10]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'Stomach_cohort.csv':
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # numerical_cols = [9, 10, 11]
        numerical_cols = [9, 10]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'in-hospital-mortality.csv':

        mean_imputing_cols = [3, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31, 32, 34, 37, 40, 44, 46, 47]
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_df.iloc[:, mean_imputing_cols] = mean_imputer.fit_transform(data_df.iloc[:, mean_imputing_cols])

        df_feas = data_df.iloc[:, 1:]
        df_target = data_df.iloc[:, 0]

        cat_cols = list(np.argwhere(df_feas.nunique().values==2).ravel())
        numerical_cols = list(np.argwhere(df_feas.nunique().values>2).ravel())

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas_num = df_feas.iloc[:, numerical_cols]
        # scaler = StandardScaler()
        # df_feas_num = scaler.fit_transform(df_feas_num)

        df_feas = pd.concat([df_feas_cat, pd.DataFrame(df_feas_num, columns=df_feas.iloc[:, numerical_cols].columns)], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'cirrhosis.csv':
        data_df.drop(['ID'], axis=1, inplace=True)
        print(data_df.isnull().sum())
        exit()
        #####
        print(data_df)
        cols = data_df.columns
        for i, col in enumerate(cols):
            print(i, col, data_df.iloc[:, i].nunique())
        exit()

        data_df.loc[data_df['bmi'].isnull(), 'bmi'] = data_df['bmi'].mean()

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [0, 2, 3, 4, 5, 6, 9]
        numerical_cols = [1, 7, 8]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'cardio_train.csv':
        data_df.drop(['id'], axis=1, inplace=True)

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [1, 6, 7, 8, 9, 10]
        numerical_cols = [0, 2, 3, 4, 5]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'adult.csv':
        data_df = data_df.replace(' ?', np.nan)

        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        data_df.loc[:, ['workclass', 'occupation', 'native-country']] = cat_imputer.fit_transform(data_df.loc[:, ['workclass', 'occupation', 'native-country']])

        cat_cols = [1, 3, 4, 5, 6, 7, 8, 9, 13]
        numerical_cols = [0, 2, 10, 11, 12]

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'Placement_Data_Full_Class.csv':
        data_df = data_df.replace(' ?', np.nan)
        data_df.loc[:, 'class'] = data_df['status']
        data_df.drop(['status', 'sl_no'], axis=1, inplace=True)
        data_df.loc[data_df['salary'].isnull(), 'salary'] = data_df['salary'].mean()

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [0, 2, 4, 5, 7, 8, 10, ]
        numerical_cols = [1, 3, 6, 9, 11, 12]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'breast-w.csv':
        data_df = data_df.replace('?', np.nan)
        data_df.loc[data_df['Bare_Nuclei'].notnull(), 'Bare_Nuclei'] = data_df.loc[data_df['Bare_Nuclei'].notnull(), 'Bare_Nuclei'].astype('int64')
        data_df.loc[data_df['Bare_Nuclei'].isnull(), 'Bare_Nuclei'] = data_df['Bare_Nuclei'].mean()

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = []
        numerical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'audit_risk.csv':
        data_df.loc[data_df['Money_Value'].isnull(), 'Money_Value'] = data_df['Money_Value'].mean()
        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [1, 3,    13, 14, 15, 16, 18, 24]
        numerical_cols = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 19, 20, 21, 22, 23, 25]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'Autism-Adult-Data.csv':
        data_df = data_df.replace('?', np.nan)
        data_df.loc[data_df['ethnicity'].isnull(), 'ethnicity'] = data_df['ethnicity'].mode().mode().values[0]
        data_df.loc[data_df['relation'].isnull(), 'relation'] = data_df['relation'].mode().values[0]
        data_df.loc[data_df['age'].notnull(), 'age'] = data_df.loc[data_df['age'].notnull(), 'age'].astype('int64')
        data_df.loc[data_df['age'].isnull(), 'age'] = data_df['age'].mean()

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18]
        numerical_cols = [10, 17, ]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'churn.csv':
        data_df.drop(['phone_number'], axis=1, inplace=True)

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        cat_cols = [0, 2, 3, 4]
        numerical_cols = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'credit-Japan.csv':
        data_df = data_df.replace('?', np.nan)

        cat_cols = [0, 3, 4, 5, 6, 8, 9, 11, 12]
        numerical_cols = [1, 2, 7, 10, 13, 14]

        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        data_df.iloc[:, cat_cols] = cat_imputer.fit_transform(data_df.iloc[:, cat_cols])
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'credit-Taiwan.csv':
        cat_cols = [1, 2, 3, 5, 6, 7, 8, 9, 10, ]
        numerical_cols = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'credit-card-econometrics.csv':
        data_df['target'] = data_df['card']
        data_df.drop(['card'], axis=1, inplace=True)

        cat_cols = [5, 6, 9]
        numerical_cols = [0, 1, 2, 3, 4, 7, 8, 10]

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'Thomas-Loan-Data.csv':

        cat_cols = [4, 5, 7, ]
        numerical_cols = [0, 1, 3, 6, 8, 9, 10, 11, 12, 13]

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        scaler = StandardScaler()
        df_feas.iloc[:, numerical_cols] = scaler.fit_transform(df_feas.iloc[:, numerical_cols])

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'hmeq.csv':
        data_df['target'] = data_df['BAD']
        data_df.drop(['BAD'], axis=1, inplace=True)

        cat_cols = [3, 4]
        numerical_cols = [0, 1, 2, 5, 6, 7, 8, 9, 10]

        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        data_df.iloc[:, cat_cols] = cat_imputer.fit_transform(data_df.iloc[:, cat_cols])
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'gmsc.csv':
        data_df['target'] = data_df['SeriousDlqin2yrs']
        data_df.drop(['SeriousDlqin2yrs'], axis=1, inplace=True)

        cat_cols = []
        numerical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target

    elif dataset_name == 'Taiwan-credit.csv':
        cat_cols = [1, 2, 3, 5, 6, 7, 8, 9, 10]
        numerical_cols = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_df.iloc[:, numerical_cols] = num_imputer.fit_transform(data_df.iloc[:, numerical_cols])

        df_feas = data_df.iloc[:, :-1]
        df_target = data_df.iloc[:, -1]

        onehot_encoder = OneHotEncoder()
        onehot_encoder = onehot_encoder.fit(df_feas.iloc[:, cat_cols])

        feas_name_list = []
        for i, col in enumerate(data_df.iloc[:, cat_cols].columns):
            for fea in onehot_encoder.categories_[i]:
                feas_name_list.append('{}-[{}]'.format(col, str(fea).replace('\'', '')))

        df_feas_cat = pd.DataFrame(onehot_encoder.transform(df_feas.iloc[:, cat_cols]).toarray(),
                                   columns=feas_name_list)

        df_feas = pd.concat([df_feas_cat, df_feas.iloc[:, numerical_cols]], axis=1)
        num_categorical_feas = df_feas_cat.shape[1]
        num_numerical_feas = len(numerical_cols)

        # Process the labels
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_target)
        df_target = pd.DataFrame(label_encoder.transform(df_target), columns=['classes'])

        df_preprocessed = pd.concat([df_feas, df_target], axis=1)  # form: df_feas_cat::df_feas_num::df_target



    return df_preprocessed, num_categorical_feas, num_numerical_feas


class ParameterDict:
    def __init__(self, baseline_parameter_dict):
        self.baseline_parameter_dict = baseline_parameter_dict

    def get_parameter_list(self, model_name):
        param_combinations = list(itertools.product(*self.baseline_parameter_dict[model_name].values()))
        param_names = list(self.baseline_parameter_dict[model_name].keys())
        param_dict_list = []
        for param_combination in param_combinations:
            param_dict_ = {}
            for i, param_name in enumerate(param_names):
                param_dict_[param_name] = param_combination[i]
            param_dict_list.append(param_dict_)
        return param_dict_list
# #####
# print(data_df)
# cols = data_df.columns
# for i, col in enumerate(cols):
#     print(i, col, data_df.iloc[:, i].nunique())
# exit()

def print_score(metrics, model_name, evaluation_recorder):
    print('The {} of {}: {}±{}'.format(metrics, model_name,
                                       np.array(evaluation_recorder.evaluation_dict[model_name][metrics]).mean(),
                                       np.array(evaluation_recorder.evaluation_dict[model_name][metrics]).std()))

def makedir(save_path):
    try:
        os.makedirs(save_path)
    except:
        pass


def record_results_summary(results_summary, evaluation_recorder, model_list, metric_list):
    for metric in metric_list:
        for model in model_list:
            results_summary.results[model][metric].append(np.array(
                evaluation_recorder.evaluation_dict[model][metric]).mean())
    return results_summary