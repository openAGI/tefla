import numpy as np
import csv
import pandas as pd
import math
from sklearn import metrics,utils
from collections import defaultdict, Counter
from sklearn.datasets import load_iris
from ast import literal_eval
from statistics import mean, mode
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper,ColumnDataSource, Whisker
from bokeh.plotting import figure, output_file, show
from bokeh.transform import transform,factor_cmap
from bokeh.io import output_notebook
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import traceback


"""helper functions
"""
def calc_acc(tp,tn,fp,fn):
    """function to calculate accuracy
    """
    try:
        acc = (tp+tn)/float(tp+tn+fn+fp)
        return round(acc,3)
    except:
        return None
    
def calc_recall(tp,fn):
    """function to calculate recall/sensitivity/true positive rate
    """
    try:
        recall = tp/float(tp+fn)
        return round(recall,3)
    except:
        return None
    
def calc_precision(tp,fp):
    """function to calculate precision
    """
    try:
        prec = tp/float(tp+fp)
        return round(prec,3)
    except:
        return None
    
def calc_specificity(tn,fp):
    """function to calculate specificity/true negative rate
    """
    try:
        spec = tn/float(tn+fp)
        return round(spec,3)
    except:
        return None
    
def calc_f1score(tp,fp,fn):
    """function to calculate f1_score
    """
    try:
        f1score = (2*tp)/float(2*tp+fn+fp)
        return round(f1score,3)
    except:
        return None
    
def calc_npv(tn,fn):
    """function to calculate negative predictive value
    """
    try:
        npv = tn/float(tn+fn)
        return round(npv,3)
    except:
        return None
    
def calc_fnr(fn,tp):
    """function to calculate false negative rate
    """
    try:
        fnr = fn/float(tp+fn)
        return round(fnr,3)
    except:
        return None
    
def calc_fpr(fp,tn):
    """function to calculate false positve rate
    """
    try:
        fpr = fp/float(tn+fp)
        return round(fpr,3)
    except:
        return None

def calc_fdr(tp,fp):
    """function to calculate false discovery rate
    """
    try:
        fdr = fp/float(tp+fp)
        return round(fdr,3)
    except:
        return None
    
def calc_for(fn,tn):
    """function to calculate false positve rate
    """
    try:
        fomr = fn/float(fn+tn)
        return round(fomr,3)
    except:
        return None
    
def calc_mcc(tp,tn,fp,fn):
    """funcrtion to calculate matthews correlation coefficient
    """
    try:
        mcc = (tp*tn-fp*fn)/float(pow((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn),0.5))
        return round(mcc,3)
    except:
        return None
def calc_kappa(truth,pred):
    """funcrtion to calculate cohen's kappa coefficient
    """
    try:
        kappa = metrics.cohen_kappa_score(truth,pred)
        return round(kappa,3)
    except:
        return None
    
def calc_mae(truth,pred):
    """function to calculate mean absolute error
    """
    try:
        mae = mean([abs(truth[i]-pred[i]) for i in range(0,len(truth))])
        return round(mae,3)
    except:
        return None
    
def calc_mse(truth,pred):
    """function to calculate mean squared error
    """
    try:
        mse = mean([pow(truth[i]-pred[i],2) for i in range(0,len(truth))])
        return round(mse,3)
    except:
        return None
def calc_rmse(truth,pred):
    """function to calculate mean squared error
    """
    try:
        rmse = pow(mean([pow(truth[i]-pred[i],2) for i in range(0,len(truth))]),0.5)
        return round(rmse,3)
    except:
        return None
def plot_conf_mat(cm,classes):
    colors = ['#d5181c', '#fdbe62', '#ffffef', '#a6e99a', '#2a9631']
    mapper = LinearColorMapper(palette=colors, low=cm.min(), high=cm.max())
    p = figure(title = "confusion matrix",x_axis_label="predicted",y_axis_label="Actual",
               x_range=[str(cls) for cls in classes], y_range=[str(cls) for cls in classes],tooltips=[("value", "@image")])
    p.image(image=[cm],x=0,y=0,dw=len(classes),dh=len(classes), palette="Spectral11")
    color_bar = ColorBar(color_mapper=mapper,location=(0, 0),ticker=BasicTicker(desired_num_ticks=len(colors)))
    p.add_layout(color_bar, 'right')
    show(p)



class evaluation():
    def __init__(self,truth_file,pred_files,eval_list,plot_list,over_all=False,ensemble_voting="soft",ensemble_weights=None):
        self.eval_result = {}
        #self.class_result = {'accuracy':nan,'precision':nan,'recall':nan,'TPR'..........}
        self.eval_list = eval_list
        self.plot_list = plot_list
        self.over_all = over_all
        if self.over_all:
            self.overall_result = {}
        self.truth_file = truth_file
        self.pred_files = pred_files
        self.ensemble_voting = ensemble_voting
        self.ensemble_weights = ensemble_weights
        self.pred = []
        self.truth = []
        self.pred_max = self.pred
        self.multilabel = False
        self.read_data()
    def read_data(self): 
        try:
            self.ensemble = len(self.pred_files)>1
            self.file_data = pd.read_csv(self.truth_file)
            if len(self.file_data.columns)<2:
                    raise Exception("At least 2 columns are required")
            for i in self.pred_files:
                pred_data =pd.read_csv(i)
                if len(pred_data.columns)<2:
                    raise Exception("At least 2 columns are required")
                self.file_data = self.file_data.merge(pred_data,on='id',how="inner")
            if len(self.file_data.columns)>3:
                self.pred = np.array([self.file_data.iloc[:,i] for i in range(2,len(self.file_data.columns))])
            else:
                self.pred = np.array(self.file_data.iloc[:,2])
            self.truth = np.array(self.file_data.iloc[:,1])
            self.ids = np.array(self.file_data.iloc[:,0])
            self.eval_list = [element.strip().lower() for element in self.eval_list]
            if type(self.truth[0])==type("str") and '[' in self.truth[0]:
                self.multilabel = True
        except Exception as e:
            print("Error: ",e)
            traceback.print_exc()
                
            
    def eval_classification(self,class_names=[]):
        try:
            if self.ensemble:
                self.pred = np.array([[literal_eval(p) for p in arr] for arr in self.pred])
                self.eval_ensemble()
            else:
                self.pred = np.array([literal_eval(p) for p in self.pred])
        
            """ if class names are not provided that means either labels are already encoded so one hot encode is not needed
                or raise exception, if it is provided then encode them 
            """
            if self.multilabel:
                    self.truth = np.array([literal_eval(p) for p in self.truth])
            if len(class_names)==0:
                if type(self.truth[0])=="str":
                    raise Exception("Either provide class names or provide encoded labels")
            else:
                self.truth = np.array([class_names.index(tval) for tval in self.truth])
            if not self.multilabel:
                if len(self.pred.shape)>1:
                    self.pred_max = np.argmax(self.pred, axis=1)
                else:
                     self.pred_max = self.pred   
                classes = list(set(np.concatenate([self.truth,self.pred_max])))
                conf_matrix = metrics.confusion_matrix(self.truth,self.pred_max, labels=classes)
                if (class_names is not None) and (len(class_names)!=0):
                    classes = [class_names[cls] for cls in classes]
                tp=[0]*len(classes)
                fp=[0]*len(classes)
                fn=[0]*len(classes)
                tn=[0]*len(classes)
                col_sum = np.sum(conf_matrix,axis=0)
                row_sum = np.sum(conf_matrix,axis=1)
                cum_sum = np.sum(conf_matrix)
                for k in range(0,len(classes)):
                    tp[k]+=conf_matrix[k,k]
                    fp[k]+=col_sum[k]-tp[k]
                    fn[k]+=row_sum[k]-tp[k]
                    tn[k]+=cum_sum-tp[k]-fp[k]-fn[k]
            else:
                mlb = MultiLabelBinarizer()
                self.truth = mlb.fit_transform(self.truth)
                self.pred = mlb.transform(self.pred)
                classes = mlb.classes_
                self.truth = self.truth*2
                np_sum = np.add(self.truth,self.pred)
                tp=[0]*len(classes)
                fp=[0]*len(classes)
                fn=[0]*len(classes)
                tn=[0]*len(classes)
                for i in range(0,len(classes)):
                    tp[i] = np.sum(np_sum[:,i]==3)
                    tn[i] = np.sum(np_sum[:,i]==0)
                    fp[i] = np.sum(np_sum[:,i]==1)
                    fn[i] = np.sum(np_sum[:,i]==2)

            """class wise evaluation"""
            for cls in classes:
                self.eval_result[cls] = {}
            for element in self.eval_list:
                if element in ['recall','true positive rate', 'sensitivity']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['recall'] = calc_recall(tp[i],fn[i])
                elif element in ['specificity', 'true negative rate']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['specificity'] = calc_specificity(tn[i],fp[i])
                elif element == 'accuracy':
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['accuracy'] = calc_acc(tp[i],tn[i],fp[i],fn[i])
                elif element in ['f1_score','f1score','fscore']:    
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['f1score'] = calc_f1score(tp[i],fp[i],fn[i])
                elif element in ['precision', 'positive predictive value','ppv']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['precision'] = calc_precision(tp[i],fp[i])
                elif element in ['negative predictive value', 'npv']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['npv'] = calc_npv(tn[i],fn[i])
                elif element in ['false negative rate', 'fnr']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['fnr'] = calc_fnr(fn[i],tp[i])
                elif element in ['false positive rate', 'fpr']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['fpr'] = calc_fpr(fp[i],tn[i])
                elif element in ['false discovery rate', 'fdr']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['fdr'] = calc_fdr(tp[i],fp[i])
                elif element in ['false omission rate', 'for']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['for'] = calc_for(fn[i],tn[i])
                elif element in ['matthews correlatin coefficient', 'mcc']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['mcc'] = calc_mcc(tp[i],tn[i],fp[i],fn[i])
                elif element in ['auc', 'area under curve']: #handle cases where tpr and fpr is not present
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['auc'] = calc_auc(self.eval_result['cls']['tpr'],self.eval_result['cls']['fpr'])
                elif element in ['kappa']:
                    for i,cls in enumerate(classes):
                        self.eval_result[cls]['kappa'] = calc_kappa(self.truth,self.pred_max)
                else:
                    raise Exception("invalid Evaluation Term")

            self.eval_plot_classification(classes,tp,fp,fn,tn)
            if self.over_all:
                self.calc_overall()
                return self.overall_result
            return self.eval_result
        except Exception as e:
            print(e)
            traceback.print_exc()
            
            
    def eval_regression(self):
        try:
            if self.ensemble:
                self.ensemble_voting = "soft"
                self.eval_ensemble()
            for element in self.eval_list:
                if element in ['mae','mean absolute error']:
                    self.eval_result['mae'] = calc_mae(self.truth,self.pred)
                elif element in ['mse','mean squared error']:
                    self.eval_result['mse'] = calc_mse(self.truth,self.pred)
                elif element in ['rmse','root mean squared error']:
                    self.eval_result['rmse'] = calc_rmse(self.truth,self.pred)
                else:
                    raise Exception("invalid Evaluation Term")
            self.eval_plot_regression()
            return self.eval_result
        except Exception as e:
            print(e)
            traceback.print_exc()
    def eval_plot_classification(self,classes,tp,fp,fn,tn,):
        try:
            output_notebook()
            for plot in self.plot_list:
                if plot in ['roc','receiver operating characteristics']:
                    if len(classes)>2:
                        for i,cls in enumerate(classes):
                            actual = [1 if t==cls else 0 for t in self.truth]
                            predicted = [p[i] for p in self.pred]
                            fpr, tpr, thresholds = metrics.roc_curve(actual, predicted)
                            p = figure(title="ROC Curve "+str(cls), x_axis_label='FPR', y_axis_label='TPR')
                            p.line(tpr, fpr, line_width=2)
                            show(p)
                    else:
                        fpr, tpr, thresholds = metrics.roc_curve(actual, predicted)
                        p = figure(title="ROC Curve", x_axis_label='FPR', y_axis_label='TPR')
                        p.line(tpr, fpr, line_width=2)
                        show(p)
                if plot in ["confusion matrix","conf matrix"]:
                    if self.multilabel:
                        for i in range(0,len(classes)):
                            cm = np.array([[tp[i],fn[i]],[fp[i],tn[i]]])
                            plot_conf_mat(cm,[classes[i],"~"+str(classes[i])])
                    else:
                        cm = metrics.confusion_matrix(self.truth,self.pred_max)
                        plot_conf_mat(cm,classes)
        except Exception as e:
                print(e)
    def eval_plot_regression(self):
        try:
            for plot in self.plot_list:
                if plot in ["residual plot"]:
                    p = figure(plot_width=800, plot_height=500)
                    p.circle(self.pred, self.pred-self.truth, size=10, color="navy", alpha=0.5)
                    show(p)
                if plot in ["error bar"]:
                    groups= ['Truth', 'Predicted']
                    mean = [np.mean(self.truth),np.mean(self.pred)]
                    sd = [np.std(self.truth),np.mean(self.pred)]
                    top = [x+e for x,e in zip(mean, sd) ]
                    down = [x-e for x,e in zip(mean, sd) ]
                    y_range = (min(down)-1,max(top)+1)
                    source = ColumnDataSource(data=dict(groups=groups, counts=mean, upper=top, lower=down))
                    p = figure(x_range=groups, plot_height=400, toolbar_location=None, title="Error Bar(sd)", y_range=y_range)
                    p.vbar(x='groups', top='counts', width=0.9, source=source, legend="groups",
                           line_color='white', fill_color=factor_cmap('groups', palette=["#972881","#186f97"],factors=groups))
                    p.add_layout(Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay"))
                    p.xgrid.grid_line_color = None
                    p.legend.orientation = "horizontal"
                    p.legend.location = "top_center"
                    show(p)
                    """Error bar with 95% confidence interval
                    """
                    total_inst = len(self.truth)
                    ci = [2*(sdev/math.sqrt(total_inst)) for sdev in sd]
                    ci_top = [x+e for x,e in zip(mean, ci) ]
                    ci_down = [x-e for x,e in zip(mean, ci) ]
                    ci_y_range = (min(ci_down)-1,max(ci_top)+1)
                    source = ColumnDataSource(data=dict(groups=groups, counts=mean, upper=ci_top, lower=ci_down))
                    p = figure(x_range=groups, plot_height=400, toolbar_location=None, title="Error Bar(95% CI)", y_range=ci_y_range)
                    p.vbar(x='groups', top='counts', width=0.9, source=source, legend="groups",
                           line_color='white', fill_color=factor_cmap('groups', palette=["#972881","#186f97"],factors=groups))
                    p.add_layout(Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay"))
                    p.xgrid.grid_line_color = None
                    p.legend.orientation = "horizontal"
                    p.legend.location = "top_center"
                    show(p)
        except Exception as e:
            print(e)

    def eval_ensemble(self):
        """this function will take multiple predictions from multiple models and will return one single prediction outpu
        according to the voting strategy(hard or soft)
        preds: list of predictions from multiple models
        voting_strategy: soft or hard
        hard voting will be used in case of classification and soft voting can be used for both classification(with probablities) and regression problems.
        """
        if self.ensemble_voting == "hard":
            final_pred = np.array([])
            for i in range(0,len(self.pred[0])):
                p_list = [int(np.argmax(p[i])) for p in self.pred]
                counts = Counter(p_list)
                p_list = sorted(p_list, key=lambda x: -counts[x])
                final_pred = np.append(final_pred, p_list[0])
            self.pred_max = final_pred
        else:
            final_pred = np.zeros_like(self.pred[0])
            if self.ensemble_weights == None or len(self.ensemble_weights) == 0:
                self.ensemble_weights = [1]*len(self.pred)
            for i in range(0,len(self.pred)):
                final_pred = np.add(final_pred,self.pred[i]*self.ensemble_weights[i])
            final_pred = final_pred/float(np.sum(self.ensemble_weights))
            if len(final_pred.shape)>1:
                self.pred_max = np.argmax(final_pred,axis=1)
        self.pred = final_pred
    def calc_overall(self):
        """function to calculate over all evaluation result
        """
        overall = defaultdict(int)

        try:
            for cls, evl in self.eval_result.items():
                for el, value in evl.items():
                    overall[el]+= value

            overall = {k:round(v/len(self.eval_result.keys()),3) for k,v in overall.items()}
            self.overall_result["overall"] = overall
        except Exception as e:
            print(e)

