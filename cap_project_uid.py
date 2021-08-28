'''Module with tools for user identification project'''
import pandas as pd
import numpy as np
import os
import dill
import time
import re
from glob import glob
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from tqdm.notebook import tqdm

PATH_TO_DATA = '..\\capstone_user_identification'
PATH_TO_FSLOG = 'features_selector_log'

tfidf_number = [0]

features_list = {
        'bag_of_sites': ['site'+str(i) for i in range(1, 11)],
        'is_top_feats': ['is_top'+str(i) for i in range(1, 31)],
        'categorical_feats': ['binned_start_hour', 'day_of_week', 'day'],
        'binary_feats': ['unique_site_1', 'unique_site_2'],
        'numerical_feats': ['unique_sites']
    }

class FeatsCounter:
    def fit(self, X, y=None):
        pass
    def transform(self, X):
        global tfidf_number
        tfidf_number[0] = X.shape[1]
        return X
    def fit_transform(self, X, y=None):
        return self.transform(X)

class FeaturesSelector:
    
    def __init__(self):
        pass
    def fit_transform(self, model, X, y, cv, blocks=None, scoring='roc_auc', n=10, continue_last=False):
        
        pipeline = create_pipeline(blocks, to_selector=True)
        self.data = pipeline.fit_transform(X)
        cat_names = []
        categories = pipeline.steps[0][1].transformer_list[2][1].steps[1][1].categories_
        for i, feat in enumerate(features_list['categorical_feats']):
            if feat=='binned_start_hour':
                cat_names.extend([feat + '_' + str(round(h.left)) for h in categories[i]])
            else:
                cat_names.extend([feat + '_' + str(el) for el in categories[i]])
        labels = features_list['binary_feats'] + cat_names + ['is_top_' + str(i) for i in range(1, 31)] +\
            features_list['numerical_feats']
        self.labels = labels
        idxs = np.array([True]*len(labels))
        scores = cross_val_score(model, self.data, y, cv=cv, scoring=scoring, n_jobs=-2)
        mean_score = scores.mean()
        delete = True
        self.report = pd.DataFrame({'feature': ['start'], 'operation': ['start'], scoring: [mean_score]})
        self.feat_values = pd.Series(0, index=labels)
        
        if continue_last:
            cur_dir = glob(os.path.join(PATH_TO_FSLOG, 'run*'))[-1]
            cur_time = re.search('run\S*', cur_dir)[0][4:]
            self.report = pd.read_csv(os.path.join(cur_dir, 'report_' + cur_time + '.csv'), index_col=0)
            self.feat_values = pd.read_csv(os.path.join(cur_dir, 'values_' + cur_time + '.csv'), 
                                           index_col=0, squeeze=True)
            with open(os.path.join(cur_dir, 'idxs.txt'), 'r') as f:
                idxs = np.array(f.read().split(' ')).astype(bool)[tfidf_number[0]:]
        else:
            cur_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
            cur_dir = os.path.join(PATH_TO_FSLOG, 'run_' + cur_time)
            os.mkdir(cur_dir)
        for i in tqdm(range(n)):
            if np.where(idxs==delete)[0].shape[0] == 0:
                delete = not delete
            pos = np.random.choice(np.where(idxs==delete)[0])
            idxs[pos] = not delete
            inner_idxs = np.hstack((np.array([True]*tfidf_number[0]), idxs))
            scores = cross_val_score(model, self.data[:, inner_idxs], y, cv=cv, scoring=scoring, n_jobs=-2)
            new_score = scores.mean()
            self.report = self.report.append(pd.DataFrame({'feature': [labels[pos]], 
                                             'operation': ['delete' if delete is True else 'add'],
                                             scoring: [new_score]}), ignore_index=True)
            self.report.to_csv(os.path.join(cur_dir, 'report_' + cur_time + '.csv'))
            add_coef = (new_score - mean_score)*(2*delete - 1)
            print(add_coef)
            self.feat_values.loc[labels[pos]] += add_coef
            self.feat_values = self.feat_values.sort_values()
            self.feat_values.to_csv(os.path.join(cur_dir, 'values_' + cur_time + '.csv'))
            if new_score < mean_score:
                idxs[pos] = not idxs[pos]
                delete = not delete
                continue
            mean_score = new_score
        #self.best_score = mean_score
        #self.best_model = create_pipeline(model=model)
        inner_idxs = np.hstack((np.array([True]*tfidf_number[0]), idxs))
        with open(os.path.join(cur_dir, 'idxs.txt'), 'w') as idxs_file:
            idxs_file.write(' '.join(inner_idxs.astype(int).astype(str)))
        self.del_labels = np.array(labels)[np.invert(idxs)]
        #self.best_model.set_params(features_filter=FunctionTransformer(lambda data: data[:, inner_idxs]))
        #with open(os.path.join(cur_dir, 'best_model.pkl'), 'wb') as best_model_pkl:
        #    dill.dump(self.best_model, best_model_pkl)

def load_catch_me_data():
    
    train_sessions = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col=0)
    order = train_sessions.time1.sort_values().index - 1
    user_id = train_sessions.loc[order+1].target
    user_id.index -= 1
    train_catch_me = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_catch_me.csv'), index_col=0)
    train_feats_catch_me = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_feats_catch_me.csv'), index_col=0)
    test_catch_me = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_catch_me.csv'), index_col=0)
    test_feats_catch_me = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_feats_catch_me.csv'), index_col=0)
    train_data = pd.concat((train_catch_me, train_feats_catch_me), axis=1).loc[order]
    test_data = pd.concat((test_catch_me, test_feats_catch_me), axis=1)
    train_data['binned_start_hour'] = pd.cut(train_data.start_hour, bins=7)
    train_data['unique_site_1'] = [1 if x==1 else 0 for x in train_data['unique_sites']]
    train_data['unique_site_2'] = [1 if x==2 else 0 for x in train_data['unique_sites']]
    test_data['binned_start_hour'] = pd.cut(test_data.start_hour, bins=7)
    test_data['unique_site_1'] = [1 if x==1 else 0 for x in test_data['unique_sites']]
    test_data['unique_site_2'] = [1 if x==2 else 0 for x in test_data['unique_sites']]
    
    return train_data, test_data, user_id

def create_pipeline(blocks=None, model=None, to_selector=False):
    
    pipeline = Pipeline(steps=[
        ('feature_processing', FeatureUnion(transformer_list=[
            ('bag_of_sites', Pipeline(steps=[
                ('selecting', FunctionTransformer(lambda data: data.loc[:, features_list['bag_of_sites']].apply(lambda row: \
                                                    ' '.join([str(x) for x in row]), axis=1))),
                ('vectorizer', TfidfVectorizer()),
                ('feats_counter', FeatsCounter())
            ])),
            ('binary_feats', FunctionTransformer(lambda data: data.loc[:, features_list['binary_feats']])),
            ('categorical_feats', Pipeline(steps=[
                ('selecting', FunctionTransformer(lambda data: data.loc[:, features_list['categorical_feats']])),
                ('hot_encoding', OneHotEncoder(handle_unknown='ignore')),
                ('dim_reducing', None)
            ])),
            ('is_top_feats', Pipeline(steps=[
                ('selecting', FunctionTransformer(lambda data: data.loc[:, features_list['is_top_feats']])),
                ('dim_reducing', None)
            ])),
            ('numerical_feats', Pipeline(steps=[
                ('selecting', FunctionTransformer(lambda data: data.loc[:, features_list['numerical_feats']])),
                ('scaling', StandardScaler())
            ])),
            ('woe_encoding', 'drop')
            ])),
        ('features_filter', None),
        ('classifier', model)
    ])
            
    if to_selector is False:
        pipeline.set_params(feature_processing__bag_of_sites__feats_counter=None)
        
    if blocks is None:
        blocks = features_list.keys()
    for block_name in features_list.keys():
        if block_name not in blocks:
            param_to_set = {'feature_processing__'+block_name: 'drop'}
            pipeline.set_params(**param_to_set)
            
    return pipeline

class WoE_Encoder():
    pass

