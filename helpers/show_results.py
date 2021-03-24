import sys
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, log_loss, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import learning_curve

sys.path.append("..")
from . import plotcfg


columns = ['Method', 'auROC', 'Accuracy', 'Detection rate', 'False alarm rate']

def fpr_rate(y_true, y_pred, **kwargs):
    tn = np.sum(((y_true == 0) & (y_pred == 0)))
    fp = np.sum(((y_true == 0) & (y_pred == 1)))
    return 1 - float(tn)/(fp+tn)

fpr_score = make_scorer(fpr_rate)

    
def bold(table):
    matrix = np.array(table)
    for j in range(1,matrix.shape[1]):
        col_values = [float(key.split('±')[0]) for key in matrix[:,j]]
        i = np.argmax(col_values) if j != 4 else np.argmin(col_values)
        matrix[i, j] = matrix[i, j].replace('  ', '**')
    return matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="Blues"):
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
             label="Training")
    plt.plot(train_sizes, test_scores_mean, color=plotcfg.tableau[1],
             label="Validation")
    


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
        plt.ylabel('Detection probability')
        plt.xlabel('False alarm probability')
        plt.grid()
    if cmGraph:
        class_names=['Ausente','Presente']
        plt.figure('cm', figsize=(6,5))    

    table = []
    
    for model_name, model in clf:
        sys.stdout.write('Processing: {:20s}'.format(model_name))       
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
                    mean_tpr += np.interp(mean_fpr, fpr, tpr)
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
    #print("Table: {{Fonte: autoria própria}}{{#tab:{}}}".format(title.lower()))
    #print()
    #print(tabulate(bold(table), headers=columns, stralign='center'))
    print()
    #print(p_table(resultados))
    return table
        
#plt.show()


