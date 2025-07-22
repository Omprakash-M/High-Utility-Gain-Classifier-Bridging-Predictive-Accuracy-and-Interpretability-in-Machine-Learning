import pandas as pd
import numpy as np
import datatable as dt
import copy
import math
import time
import subprocess
import os
import glob
from itertools import combinations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler, LabelBinarizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from scipy import io
from scipy.stats import entropy
from scipy.sparse import csr_matrix
import struct

import warnings
warnings.filterwarnings("ignore")

class HUGIMLClassifierBNB(ClassifierMixin, BaseEstimator):
    def __init__(self, 
                 allCols=None, origColumns=None, 
                 B=5, L=1, G=1e-4, 
                 dsName="unspecifiedClf", 
                 foldNo=1, 
                 base_estimator=None, 
                 imbWeights=1, 
                 huiItemsPercent=1, topK=-1, fsK=-1,
                 verbose=False):
        
        self.allCols = allCols
        self.origColumns = origColumns 
        self.B = B
        self.L = L
        self.G = G
        self.dsName = dsName
        self.foldNo = foldNo
        self.base_estimator = base_estimator
        self.imbWeights = imbWeights
        self.huiItemsPercent = huiItemsPercent
        self.topK = topK
        self.fsK = fsK
        self.verbose = verbose

    def get_column_indices(self, origColumns, column_groups):
        numInt = len(self.allCols[0])
        numIntFloat = numInt + len(self.allCols[1])
        numericIntCols_indices = [idx for idx in range(numInt)]
        numericFloatCols_indices = [idx for idx in range(numInt, numIntFloat)]
        catCols_indices = [idx for idx in range(numIntFloat, len(self.origColumns))]
        
        return [numericIntCols_indices, numericFloatCols_indices, catCols_indices]

    def write_cat_columns_to_binary(self, fname, string_array):
        with open(fname, 'wb') as f:
            for row in string_array:
                row_data = ','.join([str(r) for r in row])
                row_bytes = row_data.encode('utf-8')
                f.write(len(row_bytes).to_bytes(4, byteorder='little'))
                f.write(row_bytes)

    def read_sparse_matrix_binary(self, filename):
        with open(filename, 'rb') as fx:
            rowsCnt = int.from_bytes(fx.read(4), 'big')
            colsCnt = int.from_bytes(fx.read(4), 'big')
            nnz = int.from_bytes(fx.read(4), 'big')
            
            row_indices = np.zeros(nnz, dtype=int)
            col_indices = np.zeros(nnz, dtype=int)
            data = np.ones(nnz, dtype=int)
    
            for i in range(nnz):
                row_indices[i] = int.from_bytes(fx.read(4), 'big')
                col_indices[i] = int.from_bytes(fx.read(4), 'big')
                
        sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(rowsCnt, colsCnt))
        return sparse_matrix

    def hupTransform_test_new(self, X_test):
        base_filename = 'outputs/inpdata/'+self.dsName
        if len(self.allColsIdx[0])>0: 
            X_test[:, self.allColsIdx[0]].T.astype(np.int32).tofile(f'{base_filename}_x_test_int.bin')
        if len(self.allColsIdx[1])>0: 
            X_test[:, self.allColsIdx[1]].T.astype(np.float32).tofile(f'{base_filename}_x_test_float.bin')
        if len(self.allColsIdx[2])>0: 
            self.write_cat_columns_to_binary(f'{base_filename}_x_test_cat.bin', X_test[:, self.allColsIdx[2]].T)
        
        outputPath = 'outputs/hui/'    
        runCmd = 'java -Xms1g -Xmx6g -jar THUIsl.jar'
        runCmd += ' dsname='+self.dsName+' foldno='+str(self.foldNo)+' modeltest=true'
        runCmd += ' numRows='+str(X_test.shape[0])+" numIntCols="+str(len(self.allCols[0]))
        runCmd += " numFloatCols="+str(len(self.allCols[1]))+" numCatCols="+str(len(self.allCols[2]))
        out = subprocess.call(runCmd, shell=True, stdout=subprocess.PIPE)
        
        outputFileSparseTid = self.dsName + '_tid_sparse_test.bin'
        x_test_hup = self.read_sparse_matrix_binary(outputPath+outputFileSparseTid)
        
        if self.verbose: 
            print('X test hup size', x_test_hup.shape)
        return x_test_hup
        
    def runTopKhui_train_new(self, numRows):
        topK = self.topK
        fsK = self.fsK

        minutility = 1e10
        outputPath = 'outputs/hui/'
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        runCmd = 'java -Xms1g -Xmx6g -jar THUIsl.jar'
        runCmd += ' topK='+str(topK)+' fsK='+str(fsK)+' B='+str(self.B)+' L='+str(self.L)+' G='+str(self.G)+' dsname='+self.dsName+' foldno='+str(self.foldNo)
        runCmd += ' numRows='+str(numRows)+" numIntCols="+str(len(self.allCols[0]))+" numFloatCols="+str(len(self.allCols[1]))+" numCatCols="+str(len(self.allCols[2]))
        runCmd += ' numClasses='+str(len(self.classes_))
        out = subprocess.call(runCmd, shell=True, stdout=subprocess.PIPE)
        
        outputFileUtil = self.dsName + '_util_fs_mapped.txt'
        df = dt.fread(outputPath+outputFileUtil, sep=' ', quotechar="\'")
        assert df.shape[0]!=0, 'no HUI patterns found, change parameters and re-run'
        
        patternsMapped = df['pattern'].to_list()[0]
        
        outputFileSparseTid = self.dsName + '_tid_sparse.bin'
        x_train_hup = self.read_sparse_matrix_binary(outputPath+outputFileSparseTid)
        
        return x_train_hup, patternsMapped

    def hupTransform_train_new(self, X_train, y_train):   
        self.allColsIdx = self.get_column_indices(self.origColumns, self.allCols)

        procdata = {}
        base_filename = 'outputs/inpdata/'+self.dsName
        if len(self.allColsIdx[0])>0: 
            X_train[:, self.allColsIdx[0]].T.astype(np.int32).tofile(f'{base_filename}_x_train_int.bin')    
        if len(self.allColsIdx[1])>0: 
            X_train[:, self.allColsIdx[1]].T.astype(np.float32).tofile(f'{base_filename}_x_train_float.bin')
        if len(self.allColsIdx[2])>0: 
            self.write_cat_columns_to_binary(f'{base_filename}_x_train_cat.bin', X_train[:, self.allColsIdx[2]].T)
        y_train.T.astype(np.int32).tofile(f'{base_filename}_y_train.bin')
        
        colQuoted = np.array([f'"{col}"' for col in self.origColumns], dtype=str).reshape(-1, 1)
        self.write_cat_columns_to_binary(f'{base_filename}_allColsIdxToName.bin', colQuoted.T)
        
        if self.verbose: 
            print("starting topk hui", time.strftime("%H:%M:%S", time.localtime()))
    
        huiItemsPercent = self.huiItemsPercent
        nitems = 100
        
        lsize = {}
        for i in range(1, 7): 
            lsize[i] = 0
        for i in range(1, 7):
            lsize[i] += math.comb(nitems, i)
        
        updated=False
        if self.L==-1 or self.L==1:
            newTopK = lsize[1]
            updated=True
        elif self.L!='all' and self.L>=2 and self.L<=6:
            newTopK = huiItemsPercent*lsize[self.L]
            updated=True
        else:
            if self.topK==-1:
                newTopK = huiItemsPercent*lsize[2]
                updated = True
            else: 
                updated=False
            
        if updated:
            if self.topK==-1:
                procdata['topKoriginal'] = self.topK
                self.topK = int(newTopK)
            
            if self.fsK==-1 or self.fsK>self.topK:
                procdata['fsKoriginal'] = self.fsK
                self.fsK = self.topK
            
        x_train_hup, patternsMapped = self.runTopKhui_train_new(X_train.shape[0])
        if self.verbose: 
            print(self.topK, "actual number of itemsets generated", len(patternsMapped))

        procdata['patternsMapped'] = patternsMapped 
        procdata['x_train_hup'] = x_train_hup
        if self.verbose: 
            print('X train hup size', x_train_hup.shape)
        
        return procdata

    def get_bins(self):
        check_is_fitted(self)
        filename = 'outputs/feModels/'+self.dsName+'_kbins.bin'
        intFloatCols = len(self.allCols[0])+len(self.allCols[1])
        numBins = []
        try:
            with open(filename, 'rb') as fx:    
                for i in range(intFloatCols):
                    nb = int.from_bytes(fx.read(4), 'big')
                    fx.read(4*(nb+1))
                    numBins.append(nb)
        except IOError as e:
            print(f"Error reading file: {e}")
        return numBins
        
    def get_hug_features(self):
        check_is_fitted(self)
        return self.procdata_['patternsMapped']

    def get_transformed_shape(self):
        check_is_fitted(self)
        return self.procdata_['x_train_hup'].shape
    
    def cleanupFolderFiles(self):
        if self.foldNo==1:
            if os.path.exists('outputs'):
                time.sleep(.5)
                for f in glob.glob('outputs/'+self.dsName+'*.txt'): 
                    os.remove(f)
                for f in glob.glob('outputs/feModels/'+self.dsName+'*.txt'): 
                    os.remove(f)
                for f in glob.glob('outputs/inpdata/'+self.dsName+'*.txt'): 
                    os.remove(f)
                for f in glob.glob('outputs/inpdata/'+self.dsName+'*.bin'): 
                    os.remove(f)
                for f in glob.glob('outputs/hui/'+self.dsName+'*.txt'): 
                    os.remove(f)
                for f in glob.glob('outputs/hui/'+self.dsName+'*.bin'): 
                    os.remove(f)
    
            if not os.path.exists('outputs/'):
                os.makedirs('outputs/')
            inputPath = 'outputs/inpdata/'
            if not os.path.exists(inputPath):
                os.makedirs(inputPath)
            if not os.path.exists('outputs/feModels/'):
                os.makedirs('outputs/feModels/')

    def validateParams(self):
        assert (self.allCols!=None and self.allCols!='') and (self.origColumns!=None and self.origColumns!=''), 'Specify mandatory arguments:: \n\t\t allCols: [list of integer, list of float, list of categorical columns], \n\t\t origColumns: original column names of the data frame'
        if self.G==0: 
            self.G=0.0
        assert isinstance(self.B, int) and isinstance(self.L, int) and isinstance(self.G, float), 'give the correct type of arguments: B and L should be integers, and G should be a float value'
        
        if self.dsName=='' or (not isinstance(self.dsName, str)):
            self.dsName = 'unspecified'

    def prepareXy(self, X, y):
        assert type(X)==pd.core.frame.DataFrame, 'X should be a pandas data frame'
            
        X.columns = [str(c) for c in X.columns.tolist()]
        numericColumns = [colx for colx in X.columns.tolist() if not np.issubdtype(X[colx].dtype, object)]
        catColumns = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]

        numericIntCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, np.integer)]
        numericFloatCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, float)]
        catCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]
        
        allCols = [numericIntCols, numericFloatCols, catColumns]
        X = X[[ai for a in allCols for ai in a]]
        X = X.reset_index(drop=True)
        self.allCols = allCols
        self.origColumns = X.columns.tolist()

        assert type(y)==pd.core.series.Series, 'y should be a pandas series object'
        u, cnts = np.unique(y, return_counts=True)
        rx = sorted([(a, b) for a, b in zip(u, cnts)], key=lambda x: -x[1])
        mapYbyCntsDesc = dict([(rxi[0], ridx) for ridx, rxi in enumerate(rx)])
        y = pd.Series(y).map(mapYbyCntsDesc).to_numpy()
        self.yNewToOriginal = mapYbyCntsDesc
        
        return X, y    
    
    def fit(self, X_train, y_train):
        if type(X_train)==pd.core.frame.DataFrame:
            X_train = X_train.to_numpy()

        if type(y_train)==pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        assert type(X_train)==np.ndarray, 'X_train should be a numpy array'
        assert type(y_train)==np.ndarray, 'y_train should be a numpy array'
        
        X_train, y_train = check_X_y(X_train, y_train, dtype=None)
        
        # Changed from LogisticRegression to BernoulliNB
        if self.base_estimator==None:
            self.base_estimator = BernoulliNB()
        self.model = Pipeline([('clf', self.base_estimator)])
            
        self.n_features_in_ = X_train.shape[1]
        self.classes_ = np.unique(y_train)

        self.validateParams()
        self.cleanupFolderFiles()

        if self.verbose: 
            print('\nHUGIML Classifier with BernoulliNB')
        self.x_test_hup_ = None
        self.y_pred_proba_, self.y_pred_ = None, None

        self.procdata_ = self.hupTransform_train_new(X_train, y_train)
        if self.verbose: 
            print('transformed train shape', self.procdata_['x_train_hup'].shape)
        
        self.model.fit(self.procdata_['x_train_hup'], y_train)
        return self

    def predict_proba(self, X_test):
        if type(X_test)==pd.core.frame.DataFrame:
            X_test = X_test.to_numpy()
            
        check_is_fitted(self)
        X_test = check_array(X_test, dtype=None)

        if self.x_test_hup_==None:
            self.x_test_hup_ = self.hupTransform_test_new(X_test)
        
        self.y_pred_proba_ = self.model.predict_proba(self.x_test_hup_)
        self.y_pred_ = np.argmax(self.y_pred_proba_, axis=1)
        return self.y_pred_proba_

    def predict(self, X_test):
        if type(X_test)==pd.core.frame.DataFrame:
            X_test = X_test.to_numpy()
            
        check_is_fitted(self)
        X_test = check_array(X_test, dtype=None)

        if self.x_test_hup_==None:
            self.x_test_hup_ = self.hupTransform_test_new(X_test)
        return self.model.predict(self.x_test_hup_)