import pandas as pd
import numpy as np
import multiprocessing
from functools import partial

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from mlxtend.frequent_patterns import fpgrowth, association_rules

from utils2 import TrainedModel

class DecisionTree:
    def __init__(self, max_depth, soft_fit=False):
        self.max_depth = max_depth
        self.soft_fit = soft_fit

    def fit(self, X, y_true, y_soft):
        if not self.soft_fit:
            self.tree_discretizer = DecisionTreeClassifier(max_depth=self.max_depth)
            self.tree_discretizer.fit(X, y_true)

        elif self.soft_fit:
            y_soft = np.clip(y_soft, 1e-3, 1 - 1e-3)
            y = y_soft

            inv_sif_y = np.log(y / (1 - y))
            self.tree_discretizer = DecisionTreeRegressor(max_depth=self.max_depth)
            self.tree_discretizer.fit(X, inv_sif_y)

        self.tree_ = self.tree_discretizer.tree_

    def predict_proba(self, X):
        if not self.soft_fit:
            return self.tree_discretizer.predict_proba(X)
        elif self.soft_fit:
            y_proba_positive = sigmoid(np.clip(self.tree_discretizer.predict(X), -50, 50))
            y_proba_negative = 1 - y_proba_positive
            return np.vstack([y_proba_negative, y_proba_positive]).T

class TreeBasedDiscretizer:
    def __init__(self, number_of_categorical_features, num_numerical_features, soft_supervision=True, max_depth=2,
                 max_depth_dict=None):
        self.max_depth = max_depth
        self.soft_supervision = soft_supervision
        self.number_of_categorical_features = number_of_categorical_features
        self.num_numerical_features = num_numerical_features
        self.max_depth_dict = max_depth_dict

    def fit(self, X, y_true, y_soft):
        self.posi_rartio = float((y_true.sum() / y_true.shape[0]).iloc[0])


        X_numerical = X.iloc[:, -self.num_numerical_features:]

        self.feature_tree_dict = {}

        if self.max_depth_dict == None:
            for numerical_col in X_numerical.columns:
                tree = DecisionTree(max_depth=self.max_depth, soft_fit=self.soft_supervision)
                tree.fit(X_numerical[numerical_col].to_frame(), y_true, y_soft)
                thresholds = np.sort([i for i in tree.tree_.threshold if i != -2.])
                self.feature_tree_dict[numerical_col] = {'tree': tree, 'thresholds': thresholds}
        elif self.max_depth_dict != None:
            for numerical_col in X_numerical.columns:
                tree = DecisionTree(max_depth=self.max_depth_dict[numerical_col], soft_fit=self.soft_supervision)
                tree.fit(X_numerical[numerical_col].to_frame(), y_true, y_soft)
                thresholds = np.sort([i for i in tree.tree_.threshold if i != -2.])
                self.feature_tree_dict[numerical_col] = {'tree': tree, 'thresholds': thresholds}

    def transform(self, X):

        def generate_onehot_df(df_numerical_col, thresholds):
            def generate_interval_name(numerical_col, thresholds):
                feature_intervals = []
                feature_intervals.append('{}-[{}]'.format(numerical_col,
                                                          str([-np.inf, thresholds[0]])))

                for i in range(len(thresholds)):
                    if i < len(thresholds) - 1:
                        feature_intervals.append('{}-[{}]'.format(numerical_col,
                                                                  str([thresholds[i], thresholds[i + 1]])))
                feature_intervals.append('{}-[{}]'.format(numerical_col,
                                                          str([thresholds[-1], np.inf])))
                return feature_intervals

            def generate_interval_index(thresholds, value):
                index_ = 0
                while index_ < len(thresholds) and value >= thresholds[index_]:
                    index_ += 1
                return index_

            def generate_onehot_row(row):
                one_hot_row = [0] * (len(thresholds) + 1)
                one_hot_row[generate_interval_index(thresholds, row.iloc[0])] = 1
                return one_hot_row

            interval_names = generate_interval_name(numerical_col, thresholds)
            df_onehot = pd.DataFrame(columns=interval_names)
            df_onehot[interval_names] = df_numerical_col.apply(generate_onehot_row, axis=1, result_type="expand")

            # #####
            # for i, threshold in enumerate(thresholds):
            #     if i == 0:
            #         df_numerical_col[df_numerical_col < thresholds[i]] = str(i)
            #     elif ((i > 0) & (i < len(thresholds) - 1)):
            #         df_numerical_col[
            #             (df_numerical_col > thresholds[i]) & (df_numerical_col <= thresholds[i + 1])] = str(i)
            #     elif i == len(thresholds):
            #         df_numerical_col[df_numerical_col > thresholds[i]] = str(i)


            return df_onehot

        X_categorical = X.iloc[:, :self.number_of_categorical_features]
        X_numerical = X.iloc[:, -self.num_numerical_features:]

        X_numerical_onehot_list = []
        for numerical_col in X_numerical.columns:
            X_numerical_onehot = generate_onehot_df(df_numerical_col=X_numerical[numerical_col].to_frame(),
                                                    thresholds=self.feature_tree_dict[numerical_col]['thresholds'])
            X_numerical_onehot_list.append(X_numerical_onehot)

            # 将数值型的特征转换为树模型的预测概率
            X_numerical[numerical_col] = self.feature_tree_dict[numerical_col]['tree'].predict_proba(
                X_numerical[numerical_col].to_frame())[:, 1] - self.posi_rartio

        X_numerical_onehot_generated = pd.concat(X_numerical_onehot_list, axis=1)

        return pd.concat([X_categorical, X_numerical], axis=1), X_numerical_onehot_generated

class FeatureGenerator:
    def __init__(self, number_of_categorical_features, number_of_numerical_features,
                 max_combination_length=3, min_support=0.2, min_confidence=0.8):

        self.number_of_categorical_features = number_of_categorical_features
        self.number_of_numerical_features = number_of_numerical_features

        self.max_combination_length = max_combination_length
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit(self, X, y_true, X_generated=None, verbose=False):

        X_categorical = X.iloc[:, :self.number_of_categorical_features]
        association_mining_df = pd.concat([X_categorical, X_generated], axis=1)
        association_mining_df['classes-[1]'] = y_true
        association_mining_df['classes-[0]'] = association_mining_df['classes-[1]'].apply(lambda x: not x)
        association_mining_df = association_mining_df.astype('bool')

        # frequent itemsets and association rules
        frequent_itemsets = fpgrowth(association_mining_df, min_support=self.min_support, use_colnames=True,
                                     max_len=self.max_combination_length)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=self.min_confidence)
        self.rules_for_target_class = rules[(rules.consequents == {'classes-[1]'}) |
                                            (rules.consequents == {'classes-[0]'})].sort_values(by='lift',
                                                                                                ascending=False)
        if verbose:
            print('The number of generated feature combinations: {}'.format(len(self.rules_for_target_class.antecedents)))
            # exit()

    def transform(self, X, X_generated=None):
        X_cat = X.iloc[:, :self.number_of_categorical_features]
        X_for_mining_combinations = pd.concat([X_generated, X_cat], axis=1)

        for combined_features in self.rules_for_target_class.antecedents:
            X['{}'.format(list(combined_features))] = 0
            ## 修改这里的apply代码
            combined_feature_flag = (X_for_mining_combinations[list(combined_features)].values.sum(1)==len(combined_features))
            ## combined_feature_flag = X_for_mining_combinations.apply(lambda row: row[(list(combined_features))].sum() == len(combined_features), axis=1)
            X.loc[combined_feature_flag, '{}'.format(list(combined_features))] = 1

        X.iloc[:, :self.number_of_categorical_features] = X.iloc[:, :self.number_of_categorical_features].astype('int')
        X.iloc[:, -len(self.rules_for_target_class.antecedents):] = X.iloc[:, -len(
            self.rules_for_target_class.antecedents):].astype('int')
        return X

class LinearLearner:
    def __init__(self, soft_training, temperature, alpha):
        self.soft_training = soft_training
        self.temperature = temperature
        self.alpha = alpha

    def fit(self, X, y_true, y_soft):
        if self.soft_training:

            y_soft = np.clip(y_soft, 1e-3, 1 - 1e-3)
            y_soft_logits = np.log(y_soft / (1 - y_soft))
            y_soft = np.exp(y_soft_logits / self.temperature) / (1 + np.exp(y_soft_logits / self.temperature))
            y = self.alpha * y_soft + (1 - self.alpha) * y_true.values.ravel()

            y = np.clip(y, 1e-3, 1 - 1e-3)
            inv_sif_y = np.log(y / (1 - y))
            self.linear_model = LinearRegression()
            # self.linear_model = Lasso(alpha=0.1)

            self.linear_model.fit(X, inv_sif_y)

        elif not self.soft_training:
            self.linear_model = LogisticRegression()
            self.linear_model.fit(X, y_true)

    def predict(self, X):
        if self.soft_training:
            return np.round(sigmoid(np.clip(self.linear_model.predict(X), -50, 50)))

        elif not self.soft_training:
            return self.linear_model.predict(X)

    def predict_proba(self, X):

        if self.soft_training:
            y_proba_positive = sigmoid(np.clip(self.linear_model.predict(X), -50, 50))
            y_proba_negative = 1 - y_proba_positive
            return np.vstack([y_proba_negative, y_proba_positive]).T

        elif not self.soft_training:
            return self.linear_model.predict_proba(X)

class InterpretableAutomatedFeatureEngineering:
    def __init__(self, DTFT, ARFC, BMKD,
                 number_of_categorical_features, number_of_numerical_features,
                 TBD_max_depth_dict=None,
                 FC_min_support=None,
                 FC_min_confidence=None,
                 ST_temperature=1,
                 ST_alpha=1
                 ):

        self.discretization_method = DTFT
        self.feature_generation = ARFC
        self.soft_training = BMKD

        self.number_of_categorical_features = number_of_categorical_features
        self.number_of_numerical_features = number_of_numerical_features

        self.max_depth_dict = TBD_max_depth_dict
        self.min_support = FC_min_support
        self.min_confidence = FC_min_confidence
        self.temperature = ST_temperature
        self.alpha = ST_alpha

    def fit(self, X, y_true, y_soft):
        X_generated_in_discretization = None

        if self.discretization_method:
            self.discretizer = TreeBasedDiscretizer(number_of_categorical_features=self.number_of_categorical_features,
                                                    num_numerical_features=self.number_of_numerical_features,
                                                    max_depth_dict=self.max_depth_dict)
            self.discretizer.fit(X, y_true, y_soft)
            X, X_generated_in_discretization = self.discretizer.transform(X)

        if self.feature_generation:
            self.feature_generator = FeatureGenerator(
                number_of_categorical_features=self.number_of_categorical_features,
                number_of_numerical_features=self.number_of_numerical_features,
                min_support=self.min_support,
                min_confidence=self.min_confidence)
            self.feature_generator.fit(X=X, X_generated=X_generated_in_discretization, y_true=y_true)
            X = self.feature_generator.transform(X, X_generated_in_discretization)

        # Train a linear model
        self.linear_learner = LinearLearner(self.soft_training, self.temperature, self.alpha)
        self.linear_learner.fit(X, y_true, y_soft)

        return self

    def transform(self, X):
        X_generated_in_discretization = None
        if self.discretization_method:
            X, X_generated_in_discretization = self.discretizer.transform(X)

        if self.feature_generation:
            X = self.feature_generator.transform(X, X_generated_in_discretization)

        return X

    def predict_proba(self, X):
        return self.linear_learner.predict_proba(X)

    def predict(self, X):
        return self.linear_learner.predict(X)

def sigmoid(x):
    ex = np.exp(x)
    return ex / (1 + ex)

# def train_ILDER(X_train, y_train, y_train_soft, num_categorical_features, num_numerical_features,
#                 discretization_method=True,
#                 feature_generation=True,
#                 soft_training=True,
#                 TBD_max_depth_dict=None,
#                 FG_min_support=None,
#                 FG_min_confidence=None,
#                 ST_temperature=None,
#                 ST_alpha=None
#                 ):
#     interpretable_linear_distiller = InterpretableLinearDistiller(discretization_method=discretization_method,
#                                                                   feature_generation=feature_generation,
#                                                                   soft_training=soft_training,
#                                                                   number_of_categorical_features=num_categorical_features,
#                                                                   number_of_numerical_features=num_numerical_features,
#                                                                   TBD_max_depth_dict=TBD_max_depth_dict,
#                                                                   FG_min_support=FG_min_support,
#                                                                   FG_min_confidence=FG_min_confidence,
#                                                                   ST_temperature=ST_temperature,
#                                                                   ST_alpha=ST_alpha)
#
#     interpretable_linear_distiller.fit(X=X_train, y_true=y_train, y_soft=y_train_soft)
#
#     trained_model = TrainedModel(model_name='ILDER', trained_model=interpretable_linear_distiller)
#     return trained_model