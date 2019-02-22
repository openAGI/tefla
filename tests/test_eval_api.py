import pytest
from tefla.core.eval_metrices import evaluation

@pytest.fixture
def test_metrices():
    evaluation_list = ['accuracy','recall','precision','fpr','for','fnr','mcc','fdr','specificity','npv','f1score']
    plot_list = []
    evl = evaluation('test_data/truth_binary.csv',['test_data/pred_binary.csv'],evaluation_list,plot_list)
    res = evl.eval_classification()
    assert res[1]['accuracy']== 0.6
    assert res[1]['recall']== 0.4
    assert res[1]['precision']== 0.667
    assert res[1]['fpr']== 0.2
    assert res[1]['for']== 0.429
    assert res[1]['fnr']== 0.6
    assert res[1]['fdr']== 0.333
    assert res[1]['mcc']== 0.218
    assert res[1]['specificity']== 0.8
    assert res[1]['npv']== 0.571
    assert res[1]['f1score'] == 0.5
def test_multi():
    evaluation_list = ['accuracy','recall']
    plot_list = []
    evl = evaluation('test_data/truth_multi.csv',['test_data/pred_multi.csv'],evaluation_list,plot_list)
    res = evl.eval_classification()
    assert res[0]['accuracy']== 0.64
    assert res[0]['recall']== 0.0
    assert res[1]['accuracy']== 0.64
    assert res[1]['recall']== 0.2
    assert res[2]['accuracy']== 0.72
    assert res[2]['recall']== 0.0
    assert res[3]['accuracy']== 0.68
    assert res[3]['recall']== 0.4
    assert res[4]['accuracy']== 0.64
    assert res[4]['recall']== 0.2
    evl_2= evaluation('test_data/truth_multi.csv',['test_data/pred_multi.csv'],evaluation_list,plot_list,over_all=True)
    res_2 = evl_2.eval_classification()
    assert res_2['overall']['accuracy'] == 0.664
    assert res_2['overall']['recall'] ==0.16
def test_multi_withlabels():
    evaluation_list = ['accuracy','recall']
    plot_list = []
    evl = evaluation('test_data/truth_multi_withlabel.csv',['test_data/pred_multi_withlabel.csv'],evaluation_list,plot_list,)
    res = evl.eval_classification(['A','B','C','D','E'])
    assert res['A']['accuracy']== 0.76
    assert res['A']['recall']== 0.4
    assert res['B']['accuracy']== 0.76
    assert res['B']['recall']== 0.2
    assert res['C']['accuracy']== 0.76
    assert res['C']['recall']== 0.2
    assert res['D']['accuracy']== 0.72
    assert res['D']['recall']== 0.6
    assert res['E']['accuracy']== 0.56
    assert res['E']['recall']== 0.0
def test_multi_ensemble():
    evaluation_list = ['accuracy','recall']
    plot_list = []
    evl = evaluation('test_data/truth_multi_ens.csv',['test_data/pred_multi_ens.csv','test_data/pred_multi_ens_2.csv','test_data/pred_multi_ens_3.csv','test_data/pred_multi_ens_4.csv'],evaluation_list,plot_list,ensemble_voting="hard")
    res = evl.eval_classification()
    assert res[0]['accuracy']== 0.76
    assert res[1]['accuracy']== 0.72
    assert res[2]['accuracy']== 0.64
    assert res[3]['accuracy']== 0.68
    assert res[4]['accuracy']== 0.52
    evl_2 = evaluation('test_data/truth_multi_ens.csv',['test_data/pred_multi_ens.csv','test_data/pred_multi_ens_2.csv','test_data/pred_multi_ens_3.csv','test_data/pred_multi_ens_4.csv'],evaluation_list,plot_list,ensemble_voting="soft")
    res_2 = evl_2.eval_classification()
    assert res_2[0]['accuracy']== 0.76
    assert res_2[1]['accuracy']== 0.76
    assert res_2[2]['accuracy']== 0.56
    assert res_2[3]['accuracy']== 0.6
    assert res_2[4]['accuracy']== 0.56
def test_multilabel():
    evaluation_list = ['accuracy','recall','precision']
    plot_list = []
    evl = evaluation('test_data/truth_multilabel.csv',['test_data/pred_multilabel.csv'],evaluation_list,plot_list)
    res = evl.eval_classification()
    assert res['A']['accuracy']== 0.786
    assert res['A']['recall']== 0.667
    assert res['A']['precision']== 1.0
    assert res['B']['accuracy']== 0.857
    assert res['B']['recall']== 1.0
    assert res['B']['precision']== 0.818
    assert res['C']['accuracy']== 0.714
    assert res['C']['recall']== 0.8
    assert res['C']['precision']== 0.8
def test_reg_1():
    evaluation_list = ['mae','rmse','mse']
    plot_list = []
    evl = evaluation('test_data/truth_reg.csv',['test_data/truth_reg.csv'],evaluation_list,plot_list)
    res = evl.eval_regression()
    assert res['mae']== 0.0
    assert res['rmse']== 0.0
    assert res['mse']== 0.0
def test_reg_2():
    evaluation_list = ['mae','rmse','mse']
    plot_list = []
    evl = evaluation('test_data/truth_reg.csv',['test_data/pred_reg_1.csv'],evaluation_list,plot_list)
    res = evl.eval_regression()
    assert res['mae']== 0.546
    assert res['rmse']== 0.608
    assert res['mse']== 0.37
def test_reg_ensemble():
    evaluation_list = ['mae','rmse','mse']
    plot_list = []
    evl = evaluation('test_data/truth_reg.csv',['test/pred_reg_1.csv','test_data/pred_reg_2.csv','test/pred_reg_3.csv'],evaluation_list,plot_list)
    res = evl.eval_regression()
    assert res['mae']== 0.518
    assert res['rmse']== 0.54
    assert res['mse']== 0.291
