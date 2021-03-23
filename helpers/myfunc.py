#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, log_loss, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.signal import gaussian
import itertools
import numpy as np
import copy
from scipy import interp, signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import gzip
import pickle as pickle
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import ensemble
from statsmodels.robust import mad
import pywt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import learning_curve
from tabulate import tabulate
import sys
import time
from scipy import sparse
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import pandas as pd
from . import plotcfg

columns = ['Método', 'aROC', 'Acurácia', '$P_\\textrm{D}$', '$P_\\textrm{AF}$']

def fpr_rate(y_true, y_pred, **kwargs):
    tn = np.sum(((y_true == 0) & (y_pred == 0)))
    fp = np.sum(((y_true == 0) & (y_pred == 1)))
    return 1 - float(tn)/(fp+tn)

fpr_score = make_scorer(fpr_rate)

def p_table(results, metric='auc'):
    p_table = np.zeros((len(results), len(results)))
    for m, (modelA, pA) in enumerate(results.items()):
        for n, (modelB, pB) in enumerate(results.items()):
            t, p = stats.ttest_rel(pA[metric], pB[metric])
            p_table[m,n] = p.mean()
    columns = [x for x in results.keys()]
    index = np.array(columns).reshape(-1,1)
    print((tabulate(np.hstack((index, p_table)), numalign='center', headers=(['Tabela P'] + columns))))

    
def bold(table):
    matrix = np.array(table)
    for j in range(1,matrix.shape[1]):
        col_values = [float(key.split('±')[0]) for key in matrix[:,j]]
        i = np.argmax(col_values) if j != 4 else np.argmin(col_values)
        matrix[i, j] = matrix[i, j].replace('  ', '**')
    return matrix


def summary(title, clf, x_train, y_train, n_jobs=-1,
            train_sizes=np.linspace(.05, 1.0, 10), learnGraph=False, rocGraph=False, cmGraph=False, optimizeacc=False,
           normalize=False, x_test=None, y_test=None, cv=None, ylim=None, n_iter=10, optimize_acc=False):
    color=0
    subplot_num = 1
    resultados = {}
    if learnGraph:
        plt.figure('learn', figsize=(6,np.ceil(len(clf)/2.0)*2.5))
    if rocGraph:
        plt.figure('ROC')
        #plt.plot([0,1],[0,1], color='gray', ls='--')
        plt.xlim([0,1.0])
        plt.ylim([0,1.0])
        plt.ylabel('Probabilidade de detecção')
        plt.xlabel('Probabilidade de alarmes falsos')
        plt.grid()
    if cmGraph:
        class_names=['Ausente','Presente']
        plt.figure('cm', figsize=(6,5))    

    table = []
    
    for model_name, model in clf:
        sys.stdout.write('Processando: {:20s}'.format(model_name))       
        # medição das métricas
        start = time.time()       
        if cv is None:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:,1]
            FP = float(((y_test == 0) & (y_pred == 1)).sum())                      
            VN = float(((y_test == 0) & (y_pred == 0)).sum())            
            acuracia = accuracy_score(y_test, y_pred)
            vpositivos = recall_score(y_test, y_pred)            
            #fpositivos = 1 - VN/(FP+VN)
            fpositivos = fpr_rate(y_test, y_pred)
            
            arearoc = roc_auc_score(y_test, y_prob)
            values = [model_name] + ["  {:1.2f} ± {:1.2f}  ".format(*metric) for metric in [
                    (np.mean(arearoc), np.sqrt(np.mean(arearoc)*(1-np.mean(arearoc))/len(y_pred))),
                    (np.mean(acuracia),np.sqrt(np.mean(acuracia)*(1-np.mean(acuracia))/len(y_pred))),
                    (np.mean(vpositivos),np.sqrt(np.mean(vpositivos)*(1-np.mean(vpositivos))/len(y_pred))),
                    (np.mean(fpositivos),np.sqrt(np.mean(fpositivos)*(1-np.mean(fpositivos))/len(y_pred))),
                    ]]
            table.append(values)
                                                                                              
                                
        else:
            acuracia = []
            vpositivos = []
            fpositivos = []
            arearoc = []
            for i in range(n_iter):
                sys.stdout.write('.')
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
                acuracia.append(cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy', verbose=0, n_jobs=n_jobs))
                vpositivos.append(cross_val_score(model, x_train, y_train, cv=cv, scoring='recall', verbose=0, n_jobs=n_jobs))
                fpositivos.append(cross_val_score(model, x_train, y_train, cv=cv, scoring=fpr_score, verbose=0, n_jobs=n_jobs))
                arearoc.append(cross_val_score(model, x_train, y_train, cv=cv, scoring='roc_auc', verbose=0, n_jobs=n_jobs))                
            values = [model_name] + ["  {:1.2f} ± {:1.2f}  ".format(*metric) for metric in [
                                                                            (np.mean(arearoc), np.std(arearoc)),
                                                                            (np.mean(acuracia),np.std(acuracia)),
                                                                            (np.mean(vpositivos),np.std(vpositivos)),
                                                                            (np.mean(fpositivos),np.std(fpositivos)),                                                                            
                                                                            ]]
            
            table.append(values)
        end = time.time()
        print(' {} s'.format(end - start))
        resultados[model_name] = {'auc': arearoc,
                                'acc': acuracia,
                                 'pd': vpositivos,
                                 'paf': fpositivos,
                                 }        
        
        ##### "------------- ---------- ------------------ --------------------- ----------"        
        
        # plotação dos gráficos
        if learnGraph:
            plt.figure('learn')
            plt.subplot(np.ceil(len(clf)/2.0), 2, subplot_num)
            plt.subplots_adjust(wspace=0.3, hspace=0.4, top = 1, bottom = 0, right = 1, left = 0)
            plt.title(model_name)
            plot_learning_curve(model, x_train,y_train, cv=10, n_jobs=n_jobs, train_sizes=train_sizes,
                                ylim=ylim, test=False, label=model_name)
            plt.legend(loc='best', frameon=False)
            
            
        if rocGraph:            
            plt.figure('ROC')
            mean_tpr = 0.0
            mean_fpr = np.linspace(0,1,100)
            all_tpr = []
            best_threshold = []
            if cv is None:
                probas_ = model.fit(x_train, y_train).predict_proba(x_test)
                fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr,
                         label='{} (aROC = {:0.2f})'.format(model_name,roc_auc),
                         c=plotcfg.tableau[color])
                
            else:
                for (train, test) in cv.split(x_train, y_train):
                    probas_ = model.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
                    fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:,1])
                    accuracies = tpr*0.5 + (1-fpr)*0.5
                    best_threshold.append(thresholds[np.argmax(accuracies)])
                    mean_tpr += interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0            
                mean_tpr /= cv.n_splits
                mean_tpr[-1] = 1.0
                roc_auc = auc(mean_fpr, mean_tpr)  
                plt.plot(mean_fpr, mean_tpr,
                         label='{} (aROC = {:0.2f} ± {:0.2f})'.format(model_name,roc_auc, np.std(arearoc)),
                         c=plotcfg.tableau[color])                
                if optimize_acc:
                    print('Optimal threshold for {} is {}'.format(model_name, np.mean(best_threshold)))
        
        
        if cmGraph:
            plt.figure('cm')
            cnf_matrix = confusion_matrix(y_test,y_pred)
            #np.set_printoptions(precision=2)   
            plt.subplot(np.ceil(len(clf)/3.0), 3, subplot_num)
            plot_confusion_matrix(cnf_matrix, classes=class_names, title=model_name, normalize=normalize)
        
        color += 1
        subplot_num += 1

    if rocGraph:
        plt.figure('ROC')        
        plt.legend(loc='lower right', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig('ROC-{}.pdf'.format(title), bbox_inches='tight', pad_inches = 0)        
    
    if learnGraph:
        plt.figure('learn')        
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()
        plt.savefig('learn-{}.pdf'.format(title), bbox_inches='tight', pad_inches = 0)
            
    if cmGraph:
        plt.figure('cm')        
        #plt.legend(loc='best', frameon=False)
        plt.subplots_adjust(wspace=0.5, hspace=0., top = 1, bottom = 0, right = 1, left = 0)
        #plt.tight_layout()
        plt.savefig('cm-{}.pdf'.format(title), bbox_inches='tight', pad_inches = 0)
    
    print()
    print("Table: {{Fonte: autoria própria}}{{#tab:{}}}".format(title.lower()))
    print()
    print(tabulate(bold(table), headers=columns, stralign='center'))
    print()
    print(p_table(resultados))
    return resultados
        
#plt.show()
        
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)#, rotation=45)
    plt.yticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm * 100.,1)
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Classe real')
    plt.xlabel('Classe prevista')


def snip(y, iter=8, fwhm=3, mode='fwhm'):
    y = y.astype('float')    
    channels = len(y)    
    v = np.log(np.log(np.sqrt(y+1)+1)+1)
    w = int(round(2*fwhm,0))
    for i in range(1,iter):
        if mode=='fwhm':
            mask = np.zeros(2*w+1)
        elif mode=='iter':
            mask = np.zeros(2*i+1)
        mask[0] = 0.5
        mask[-1] = 0.5       
        avgwindow = np.convolve(v, mask, mode='same')
        v = np.minimum(v,avgwindow)
    #ynew = (np.exp(np.exp(v)-1)-1)**2 - 1
    ynew = np.exp(np.exp(v)) * (np.exp(np.exp(v)) - 2*np.exp(1))/(np.exp(1)**2)
    return ynew

class Cwt(BaseEstimator,TransformerMixin):
    def __init__(self, maxwidth=5):
        self.maxwidth = maxwidth
    def transform(self, X, *_):
        widths = np.linspace(1,8,15)/2.355
        new_X = []
        for i in range(len(X)):            
            new_X.append(signal.cwt(X[i], signal.ricker, widths).ravel())
        return np.array(new_X)
    def fit(self, X, *_):
        return self

class ColumnExtractor(BaseEstimator,TransformerMixin):
    def __init__(self, colrange=(135,138)):
        self.colrange = colrange
    def transform(self, X, *_):
        cols = X[:,self.colrange[0]:self.colrange[-1]] # column 3 and 4 are "extracted"
        return cols
    def fit(self, X, *_):
        return self
    

class CustomColumns(BaseEstimator,TransformerMixin):
    def __init__(self, ids=None):
        self.ids = ids
    def transform(self, X, *_):
        return X[:, self.ids]
    def fit(self,X,*_):
        return self
    
class Normalize(BaseEstimator,TransformerMixin):
    def __init__(self, func=np.std):
        self.func = func
    def transform(self, X, *_):
        #print np.array([row/self.func(row) for row in X])
        return np.array([row/self.func(row) for row in X])
    def fit(self,X,*_):
        return self  
    
class Convolve(BaseEstimator,TransformerMixin):
    def __init__(self, fwhm=1.17, peakwidth=5, continuum=11, modulate=False):
        self.fwhm = fwhm
        self.peakwidth = peakwidth
        self.continuum = continuum
        self.modulate = modulate
    def transform(self, X, *_):
        return convolve(X, fwhm=self.fwhm, peakwidth=self.peakwidth,
                        continuum=self.continuum, modulate=self.modulate, start_from=0)
    def fit(self, X, *_):
        return self
    
class Gradient(BaseEstimator,TransformerMixin):
    def __init__(self, order=1):
        self.order = order        
    def transform(self, X, *_):
        for i in range(len(X)):
            X[i] = np.gradient(X[i], edge_order=self.order)
        return X
    def fit(self, X, *_):
        return self
    
class WaveletDenoise(BaseEstimator,TransformerMixin):
    def __init__(self, wavelet='db8', level=4, span=8):
        self.wavelet = wavelet
        self.level = level
        self.span = span
    def transform(self, X, *_):
        for i in range(len(X)):
            noisy_coefs = pywt.wavedec(X[i], self.wavelet, level=self.level, mode='per')     
            sigma = mad(noisy_coefs[-1])
            uthresh = sigma*np.sqrt(2*np.log(len(X[i])))/np.sqrt(self.span)
            denoised = noisy_coefs[:]
            denoised[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in denoised[1:])            
            X[i] = pywt.waverec(denoised, self.wavelet, mode='per')
        return X
    def fit(self, X, *_):
        return self
    
    
def convolve_array(array, fwhm=1.17, peakwidth=5, continuum=11, modulate=False):
    '''
    Convolves a matrix in the form [[y x1 x2 ... xn],...] using a gaussian peak mask (DON'T USE IT ON ARRAYS ALONE)
    '''
    sigma = fwhm/2.355
    gauss = gaussian(peakwidth,sigma)
    lefttail = int(np.ceil((continuum - peakwidth)/2.))
    righttail = int(lefttail + (continuum - peakwidth)%2.)
    sumGauss = -gauss.sum()/(continuum - peakwidth)
    mask = np.hstack(([sumGauss]*lefttail, gauss, [sumGauss]*righttail))
    array = np.array(array).astype('float')
    convolved_array = np.convolve(array, mask, mode='same')
    if modulate:
        convolved_array[convolved_array < 0 ] = 0.01    
    return convolved_array

def convolve(matrix, fwhm=1.17, peakwidth=5, continuum=11, modulate=False, start_from=1):
    '''
    Convolves a matrix in the form [[y x1 x2 ... xn],...] using a gaussian peak mask (DON'T USE IT ON ARRAYS ALONE)
    '''    
    convolved_matrix = copy.deepcopy(matrix)
    convolved_matrix = convolved_matrix.astype('float')
    for i in range(len(matrix)):
        convolved_matrix[i,start_from:] = convolve_array(convolved_matrix[i,start_from:],
                                                         fwhm,
                                                         peakwidth,
                                                         continuum,
                                                         modulate)
        #convolved_matrix[i,start_from:] = np.convolve(convolved_matrix[i,start_from:], mask, mode='same')
        #if modulate:
       #     convolved_matrix[i,start_from:][convolved_matrix[i,start_from:] < 0 ] = 0.1
        #if normalize:
        #    max_height = np.max(convolved_matrix[i,start_from:])
        #    print max_height
       #     convolved_matrix[i,start_from:] /= max_height
    return convolved_matrix


class Traditional(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, method="deriv", test_samples=[0]):
        """
        Called when initializing the classifier
        """
        self.method = method
        self.test_samples = test_samples
        self.filename = "{}_array.p".format(self.method)
        self.scores = pickle.load(gzip.open(self.filename, 'rb'))
        
    def fit(self, X=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """      

        return self

    def _threshold(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        threshold = 0.05 if self.method == 'corrbib' else 3.0
        return (1.0 if x >= threshold else 0.0)

    def predict(self, X=None, y=None):        
        return np.array([self._threshold(x) for x in self.scores[self.test_samples]])
    
    def predict_proba(self, X=None, y=None):
        scores_pos = self.scores[self.test_samples]/np.max(self.scores[self.test_samples])
        scores_neg = 1 - scores_pos
        scores = np.hstack((scores_neg.reshape(-1,1), scores_pos.reshape(-1,1) ))
        return scores

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X))) 


# In[12]:



    
    


    

class LibCorNID(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""
    def __init__(self, channel=142, fwhm=3, sensitivity=0.53, tol=1):
        """
        Called when initializing the classifier
        """
        self.channel = channel
        self.fwhm = fwhm
        self.sensitivity = sensitivity
        self.tol = tol
        
    def fit(self, X=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """      

        return self
    
    def _erosion(self, x, iterations=9):
        for n in range(1,iterations):
            m = int(round(3.0/n*self.fwhm,0))
            mask = np.zeros(2*m + 1)
            mask[0] = 0.5
            mask[-1] = 0.5        
            avg = np.convolve(x, mask, mode='same')            
            x = np.minimum(x,avg+2*np.sqrt(avg))
        return x
    
    def _gauss(self, x,a,x0):
        return a*np.exp(-(x-x0)**2/(2*(self.fwhm/2.355)**2))
    
    def _corr(self, x):
        transformed = x - self._erosion(x)        
        m = int(round(1.5*self.fwhm, 0))
        window = np.arange(self.channel-m, self.channel+m+1)
        count = x[self.channel-m:self.channel+m+1]          
        try:
            popt,_ = curve_fit(self._gauss,window,count,p0=[np.sum(count), self.channel])
            dist = abs(popt[0] - self.channel)
            #corr = abs(np.corrcoef(count, self._gauss(window, *popt)))[0][1]
            corr = np.corrcoef(count, self._gauss(window, *popt))[0][1]
            corr = 0 if corr < 0 else corr
        except RuntimeError:
            corr = 0       
        return corr
        

    def predict(self, X, y=None):
        prediction = [self._corr(x) > self.sensitivity for x in X]
        return np.array(prediction).astype('float')
    
    def predict_proba(self, X, y=None):
        scores_pos = [self._corr(x) for x in X]        
        scores_pos = np.array(scores_pos)
        scores_neg = 1 - scores_pos
        scores = np.hstack((scores_neg.reshape(-1,1), scores_pos.reshape(-1,1) ))
        return scores

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X)))
    
    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self


    
    
class Manual(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""
    def __init__(self, channel=142, fwhm=3, alpha=0.05, l=4, fwhm_multiplier=1.25):
        """
        Called when initializing the classifier
        """
        self.channel = channel
        self.fwhm = fwhm
        self.alpha = alpha
        self.l = l
        self.fwhm_multiplier = fwhm_multiplier
        
    def fit(self, X=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """      

        return self
    
    def _prob(self, x):
        m = int(round(self.fwhm*self.fwhm_multiplier, 0))        
        b = 2*m+1
        N1 = np.sum(x[self.channel-m-self.l:self.channel-m])
        N2 = np.sum(x[self.channel+m+1:self.channel+m+self.l+1])
        Ns = np.sum(x[self.channel-m:self.channel+m+1])            
        N0 = (N1 + N2)*b/(2.0*self.l)
        Nn = Ns - N0
        #print Ns, N0
        sigma_Nn = np.sqrt(Ns + N0*b/(2*self.l))
        prob = stats.norm.cdf(Nn, 0, sigma_Nn)
        return prob
    
    def predict_proba(self, X, y=None):        
        scores_pos = [self._prob(x) for x in X]        
        scores_pos = np.array(scores_pos)
        scores_neg = 1 - scores_pos
        scores = np.hstack((scores_neg.reshape(-1,1), scores_pos.reshape(-1,1) ))
        return scores        

    def predict(self, X, y=None):
        prediction = [self._prob(x) > (1-self.alpha) for x in X]
        return np.array(prediction).astype('float')   

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X)))   
    
    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self



class Erosion(BaseEstimator,TransformerMixin):
    def __init__(self, iterations=9, fwhm=3):
        self.iterations = iterations
        self.fwhm = fwhm
    def _erosion(self, x):
        for n in range(1,self.iterations):
                m = int(round(3.0/n*self.fwhm,0))
                mask = np.zeros(2*m + 1)
                mask[0] = 0.5
                mask[-1] = 0.5        
                avg = np.convolve(x, mask, mode='same')        
                x = np.minimum(x,avg+2*np.sqrt(avg))
        return x
    def transform(self, X, *_):
        for i in range(len(X)):
            X[i] = X[i] - self._erosion(X[i])
        return X
    def fit(self, X, *_):
        return self
    

    
def snip(y, iter=8, fwhm=3, mode='fwhm', convmode='same'):
    y = y.astype('float')    
    channels = len(y)    
    v = np.log(np.log(np.sqrt(y+1)+1)+1)
    w = int(round(2*fwhm,0))
    for i in range(1,iter):
        if mode=='fwhm':
            mask = np.zeros(2*w+1)
        elif mode=='iter':
            mask = np.zeros(2*i+1)
        mask[0] = 0.5
        mask[-1] = 0.5       
        avgwindow = np.convolve(v, mask, mode=convmode)
        v = np.minimum(v,avgwindow)
    #ynew = (np.exp(np.exp(v)-1)-1)**2 - 1
    ynew = np.exp(np.exp(v)) * (np.exp(np.exp(v)) - 2*np.exp(1))/(np.exp(1)**2)
    return ynew

class Snip(BaseEstimator,TransformerMixin):
    def __init__(self, iterations=8, fwhm=3, mode='iter'):
        self.iterations = iterations
        self.fwhm = fwhm
    def transform(self, X, *_):
        for i in range(len(X)):
            X[i] = X[i] - snip(X[i], iter=self.iterations, fwhm=self.fwhm, mode=self.mode)
        return X
    def fit(self, X, *_):
        return self


# In[13]:


def plot_learning_curve(estimator, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.05, 1.0, 10), test=True, label=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    #plt.figure()    
    if ylim is not None:
        plt.ylim(*ylim)
    #plt.xlim((0,1000))
    #plotcfg.hide_spines()
    #plt.tick_params(axis="both", which="both", bottom="off", top="off",    
    #            labelbottom="on", left="off", right="off", labelleft="on")
    plt.xlabel("Número de amostras")
    plt.ylabel("Erro")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='accuracy',
        n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = 1-np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = 1-np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color=plotcfg.tableau[0])
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color=plotcfg.tableau[1])
    plt.plot(train_sizes, train_scores_mean, color=plotcfg.tableau[0],
             label="Treino")
    plt.plot(train_sizes, test_scores_mean, color=plotcfg.tableau[1],
             label="Validação")
    


# In[14]:


class Haar(BaseEstimator, TransformerMixin):
    def __init__(self, spectrum='cA'):
        self.spectrum=spectrum
    def transform(self, X, *_):
        result = []     
        for rowdata in X:
            cA, cD = pywt.dwt(rowdata, 'haar')
            if self.spectrum == 'cD':
                result.append(cD)
            else:
                result.append(cA)
            
        return result
    
    def fit(self, *_):
        return self
    
def erosion(x, iterations=9, fwhm=3):
        for n in range(1,iterations):
            m = int(round(3.0/n*fwhm,0))
            mask = np.zeros(2*m + 1)
            mask[0] = 0.5
            mask[-1] = 0.5        
            avg = np.convolve(x, mask, mode='same')            
            x = np.minimum(x,avg+2*np.sqrt(avg))
        return x


# In[15]:


def baseline_als(y, lam=1e6, p=0.4, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L),2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p*(y > z) + (1-p)*(y < z)
    return z

class BaselineRemover(BaseEstimator,TransformerMixin):
    def __init__(self, lam=1e7, p=0.4, niter=3):
        self.lam = lam
        self.p = p
        self.niter = niter
    def transform(self, X, *_):
        result = []     
        for rowdata in X:
            rowdata = rowdata - baseline_als(rowdata, lam=self.lam,
                                             p=self.p, 
                                             niter=self.niter)
            result.append(rowdata)
                    
        return result
    
    def fit(self, *_):
        return self


# In[16]:


class Correlation(BaseEstimator,TransformerMixin):
    def __init__(self, p=0.05):        
        self.p = p        
    def transform(self, X, *_):                          
        return X[:,self.positions_]
    
    def fit(self, X, y, *_):
        select = SelectKBest(f_classif, 'all')
        select.fit(X,y)
        self.pvalues_ = select.pvalues_
        self.positions_ = self.pvalues_ < self.p
        return self
    
class ForestSelect(BaseEstimator,TransformerMixin):
    def __init__(self, k=21, trees=500, max_depth=4):        
        self.k = k
        self.trees = trees
        self.max_depth = max_depth
    def transform(self, X, *_):                          
        return X[:,self.positions_]
    
    def fit(self, X, y, *_):
        select = ensemble.RandomForestClassifier(self.trees, max_depth=self.max_depth)
        select.fit(X,y)
        self.importances_ = select.feature_importances_
        self.positions_ = np.argsort(self.importances_)[::-1][:self.k]
        return self


# In[17]:


def medianbg(x, window=100):
    iterations = int(len(x)/int(window))
    bg = np.zeros(len(x))
    for i in range(iterations):
        if i + 1 > iterations:
            bg[i*window:] = np.median(x[i*window:])
        else:
            bg[i*window:(i+1)*window] = np.median(x[i*window:(i+1)*window])
    return bg
        
class MedianRemover(BaseEstimator,TransformerMixin):
    def __init__(self, window=100):
        self.window = window
    def transform(self, X, *_):
        return np.array([x - medianbg(x, self.window) for x in X])
    
    def fit(self, *_):
        return self    


# In[18]:


class Log(BaseEstimator,TransformerMixin):
    def __init__(self, offset=0):
        self.offset = offset
    def transform(self, X, *_):
        return np.array([np.log(x + self.offset) for x in X])
    
    def fit(self, *_):
        return self    


# In[19]:


class SecondDifference(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, channel=142, fwhm=3, threshold=-1.47, tol=1, pdf=False):
        """
        Called when initializing the classifier
        """
        self.channel = channel
        self.fwhm = fwhm
        self.threshold = threshold
        self.p = self.fwhm/2.355
        self.tol = tol
        self.pdf = pdf
        
    def fit(self, X=None, y=None):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier

        return self
    
    def _seconddiff(self, x, y=None):        
        c = [-100]
        j = 0
        new_c = lambda p, j: 100*(j**2 - p**2)/p**2 * np.exp(-0.5 * j**2/p**2) 
        for iteration in range(50):
            j += 1         
            c.append(new_c(self.p,j))
            if abs(new_c(self.p,j+1)) < 1:
                break
        c = np.array(c)
        c[1] = -np.sum(c) + c[1]
        c[1:] /= 2.0      
        c = np.hstack((c[::-1], c[1:]))
        #print c.sum()
        sd = np.sqrt(np.convolve(x, c**2, mode='same'))
        dd = np.convolve(x, c, mode='same')
        ss = dd/sd
        #plt.plot(c)
        return ss

    def predict(self, X, y=None):
        prediction = []        
        for x in X:
            transformed = self._seconddiff(x)
            peaklist = np.argwhere(transformed <= self.threshold)
            distances = abs(self.channel - peaklist)
            #print distances
            if (len(distances) > 0):
                if np.min(distances) <= self.tol:
                    prediction.append(1.0)
                else:
                    prediction.append(0.0)
            else:
                prediction.append(0.0)
        return np.array(prediction)
    
    def _prob(self, sig):
        if self.pdf:
            return stats.norm.cdf(sig**2, self.threshold**2, -self.threshold)
        else:
            return -sig
    
    def predict_proba(self, X, y=None):
        scores_pos = []        
        for x in X:
            # espectro transformado
            transformed = self._seconddiff(x)
            # canais onde ss é menor que threshold
            peaklist = np.argwhere(transformed <= self.threshold)
            # distância entre os canais onde ss é menor que threshold e o canal onde devia estar o pico
            distances = abs(self.channel - peaklist)
            # ss dos picos cuja distância do pico alvo eh menor que tol
            peaks_in_tol = transformed[peaklist[distances <= self.tol]]            
            # média das ss dos picos cuja distância do pico alvo é menor que tol
            if len(peaks_in_tol) > 0:
                mean_ss = np.mean(peaks_in_tol)
            else:
                mean_ss = transformed[self.channel]
            scores_pos.append(self._prob(mean_ss))            
        scores_pos = np.array(scores_pos)
        scores_neg = 1 - scores_pos
        scores = np.hstack((scores_neg.reshape(-1,1), scores_pos.reshape(-1,1) ))
        return scores

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X) == y)/float(len(y)))
    
    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self


# In[ ]:




