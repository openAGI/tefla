"""
  Evaluation API for evaluation matrices and plots, supports classification and regression both.
"""

import math
from ast import literal_eval
from collections import defaultdict, Counter
from statistics import mean
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, Whisker
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from tefla.utils import quadratic_weighted_kappa as qwk


def calc_acc(true_pos, true_neg, false_pos, false_neg):
  """
    function to calculate accuracy.
  Args:
    true_pos: Number of true positives
    true_neg: Number of true negatives
    false_pos: Number of false positives
    false_neg: Number of false negatives
  Returns:
    None
  """
  try:
    acc = (true_pos + true_neg) / \
        float(true_pos + true_neg + false_neg + false_pos)
    return round(acc, 3)
  except BaseException:
    return None


def calc_recall(true_pos, false_neg):
  """
    function to calculate recall/sensitivity/true positive rate
  Args:
    true_pos: Number of true positives
    false_pos: Number of false positives
  Returns:
    None
  """
  try:
    recall = true_pos / float(true_pos + false_neg)
    return round(recall, 3)
  except BaseException:
    return None


def calc_precision(true_pos, false_pos):
  """
    function to calculate precision
  Args:
    true_pos: Number of true positives
    false_pos: Number of false positives
  Returns:
    None
  """
  try:
    prec = true_pos / float(true_pos + false_pos)
    return round(prec, 3)
  except BaseException:
    return None


def calc_specificity(true_neg, false_pos):
  """
    function to calculate specificity/true negative rate
  Args:
    true_neg: Number of true negatives
    false_pos: Number of false positives
  Returns:
    None
  """
  try:
    spec = true_neg / float(true_neg + false_pos)
    return round(spec, 3)
  except BaseException:
    return None


def calc_f1score(true_pos, false_pos, false_neg):
  """
    function to calculate f1_score
  Args:
    true_pos: Number of true positives
    false_pos: Number of false positives
    false_neg: Number of false negatives
  Returns:
    None
  """
  try:
    f1score = (2 * true_pos) / float(2 * true_pos + false_neg + false_pos)
    return round(f1score, 3)
  except BaseException:
    return None


def calc_npv(true_neg, false_neg):
  """
    function to calculate negative predictive value
  Args:
    true_neg: Number of true negatives
    false_neg: Number of false negatives
  Returns:
    None
  """
  try:
    npv = true_neg / float(true_neg + false_neg)
    return round(npv, 3)
  except BaseException:
    return None


def calc_fnr(false_neg, true_pos):
  """
    function to calculate false negative rate
  Args:
    true_pos: Number of true positives
    false_neg: Number of false negatives
  Returns:
    None
  """
  try:
    fnr = false_neg / float(true_pos + false_neg)
    return round(fnr, 3)
  except BaseException:
    return None


def calc_fpr(false_pos, true_neg):
  """
    function to calculate false positve rate
  Args:
    true_neg: Number of true negatives
    false_pos: Number of false positives
  Returns:
    None
  """
  try:
    fpr = false_pos / float(true_neg + false_pos)
    return round(fpr, 3)
  except BaseException:
    return None


def calc_fdr(true_pos, false_pos):
  """
    function to calculate false discovery rate
  Args:
    true_pos: Number of true positives
    false_pos: Number of false positives
  Returns:
    None
  """
  try:
    fdr = false_pos / float(true_pos + false_pos)
    return round(fdr, 3)
  except BaseException:
    return None


def calc_for(false_neg, true_neg):
  """
    function to calculate false positve rate
  Args:
    true_neg: Number of true negatives
    false_neg: Number of false negatives
  Returns:
    None
  """
  try:
    fomr = false_neg / float(false_neg + true_neg)
    return round(fomr, 3)
  except BaseException:
    return None


def calc_mcc(true_pos, true_neg, false_pos, false_neg):
  """
    funcrtion to calculate matthews correlation coefficient
  Args:
    true_pos: Number of true positives
    true_neg: Number of true negatives
    false_pos: Number of false positives
    false_neg: Number of false negatives
  Returns:
    None
  """
  try:
    temp_var1 = (true_pos * true_neg - false_pos * false_neg)
    temp_var2 = math.sqrt((true_pos + false_pos) * (true_pos + false_neg)
                          * (true_neg + false_pos) * (true_neg + false_neg))
    mcc = temp_var1 / float(temp_var2)
    return round(mcc, 3)
  except BaseException:
    return None


def calc_kappa(truth, pred):
  """
    funcrtion to calculate cohen's kappa coefficient
  Args:
    truth: Collections of truth labels
    pred: Collections of prediction labels
  Returns:
    None
  """
  try:
    kappa = qwk.calculate_kappa(truth, pred)
    return round(kappa, 3)
  except BaseException:
    return None


def calc_mae(truth, pred):
  """
    function to calculate mean absolute error
  Args:
    truth: Collections of truth labels
    pred: Collections of prediction labels
  Returns:
    None
  """
  try:
    mae = mean([abs(truth[i] - pred[i]) for i in range(0, len(truth))])
    return round(mae, 3)
  except BaseException:
    return None


def calc_mse(truth, pred):
  """
    function to calculate mean squared error
  Args:
    truth: Collections of truth labels
    pred: Collections of prediction labels
  Returns:
    None
  """
  try:
    mse = mean([pow(truth[i] - pred[i], 2) for i in range(0, len(truth))])
    return round(mse, 3)
  except BaseException:
    return None


def calc_rmse(truth, pred):
  """
    function to calculate mean squared error
  Args:
    truth: Collections of truth labels
    pred: Collections of prediction labels
  Returns:
    None
  """
  try:
    rmse = pow(mean([pow(truth[i] - pred[i], 2) for i in range(0, len(truth))]), 0.5)
    return round(rmse, 3)
  except BaseException:
    return None


def plot_conf_mat(conf_mat, classes):
  """
    This method plots confusion matrix.
  Args:
    conf_mat: Cnfusion matrix array
    classes: Class list
  Returns:
    bokeh plot object
  """
  mapper = LinearColorMapper(palette='Blues9', low=conf_mat.min(), high=conf_mat.max())
  fig = figure(
      title="confusion matrix",
      x_axis_label="predicted",
      y_axis_label="Actual",
      x_range=[str(cls) for cls in classes],
      y_range=[str(cls) for cls in classes],
      tooltips=[("value", "@image")])
  fig.image(image=[conf_mat], x=0, y=0, dw=len(classes), dh=len(classes), palette='Blues9')
  color_bar = ColorBar(
      color_mapper=mapper, location=(0, 0), ticker=BasicTicker(desired_num_ticks=9))
  fig.add_layout(color_bar, 'right')
  return fig


class Evaluation():
  """
    This is a base class for evaluation.
  """
  # pylint: disable=too-many-instance-attributes

  def __init__(self):
    self.eval_result = {}
    self.ids = []
    self.pred = []
    self.truth = []
    self.pred_max = self.pred
    self.ensemble = False
    self.multilabel = False
    self.eval_plots = []
    self.classes = []

  def read_data(self, truth_file, pred_files):
    """
      This method reads the data from provided csv files for truth and predication.
    Args:
      truth_file: A file containing ids and truth annotations.
      pred_files: A list of files containing ids and prediction annotations.
    Raises:
      At least 2 columns are required, if number of columns
      in the provided csv file are less then 2.
    Return:
      None.
    """
    self.ensemble = len(pred_files) > 1
    file_data = pd.read_csv(truth_file)
    if len(file_data.columns) < 2:
      raise ValueError("At least 2 columns are required")
    for i in pred_files:
      pred_data = pd.read_csv(i)
      if len(pred_data.columns) < 2:
        raise ValueError("At least 2 columns are required")
      if len(pred_data.columns) > 2:
        self.classes = pred_data.columns[1:].tolist()
        pred_data['Pred'] = pred_data[pred_data.columns[1:]].replace(
            '', 0).stack().groupby(level=0).apply(list)
        pred_data = pred_data[[pred_data.columns[0], 'Pred']]
      file_data = file_data.merge(pred_data, on=file_data.columns[0], how="inner")
    if len(file_data.columns) > 3:
      self.pred = np.array([file_data.iloc[:, i] for i in range(2, len(file_data.columns))])
    else:
      self.pred = np.array(file_data.iloc[:, 2].tolist())
    self.truth = np.array(file_data.iloc[:, 1])
    self.ids = np.array(file_data.iloc[:, 0])
    if isinstance(self.truth[0], type("str")) and '[' in self.truth[0]:
      self.multilabel = True
    else:
      self.multilabel = False

# pylint: disable-msg=too-many-arguments
# pylint: disable-msg=too-many-locals
# pylint: disable-msg=too-many-statements
# pylint: disable-msg=too-many-branches

  def eval_classification(self,
                          truth_file,
                          pred_files,
                          eval_list,
                          plot_list,
                          over_all=False,
                          ensemble_voting="soft",
                          ensemble_weights=None,
                          class_names=None,
                          convert_binary=False,
                          binary_threshold=None):
    """
      This function calculates the evaluation measures
      required by the user for classification problems.
    Args:
      truth_file: A csv file containing ids and truth annotations.
      pred_files: A list of csv files containing ids and prediction annotations.
      eval_list: A list of evaluation measures.
      plot_list: A list of evaluation plots.
      over_all: if class wise results are required or overall result,
      in case of multiclass classification, default false.
      ensemble_voting: Type of voting in case of multiple prediction(soft/hard) default soft.
      ensemble_weights: Weights for each class in case of ensemble, default None.
      calss_names: An array containing class names, default None.
      convert_binary: If multiclass predictions should be evaluated as binary problem
                      (normal vs abnormal)first value should represent probability of normal class.
      binary_threshold: threshold to be used in case of multiclass to binary conversion.
    Returns:
      A dictionary containing evaluation result.
    Raises:
      "Invalid evaluation term" if a term is not
      present in supported list.
    """
    self.eval_result = {}
    self.classes = None
    self.eval_plots = []
    self.read_data(truth_file, pred_files)
    eval_list = [element.strip().lower() for element in eval_list]
    if self.ensemble:
      self.eval_ensemble(ensemble_voting, ensemble_weights)
    if self.multilabel:
      self.truth = np.array([literal_eval(p) for p in self.truth])
      self.pred = np.array([literal_eval(p) for p in self.pred])
    classes = self.classes
    if class_names:
      classes = class_names
    if isinstance(self.truth[0], type("str")):
      self.truth = np.array([classes.index(tval) for tval in self.truth])
    if not self.multilabel:
      if len(self.pred.shape) > 1 and convert_binary and binary_threshold:
        self.pred_max = np.array([0 if prd_n[0] >= binary_threshold else 1 for prd_n in self.pred])
        self.truth = np.array([1 if truth_n != 0 else truth_n for truth_n in self.truth])
        classes = [self.classes[0], '!' + self.classes[0]]
      elif len(self.pred.shape) > 1:
        self.pred_max = np.argmax(self.pred, axis=1)
      else:
        self.pred_max = self.pred
      conf_matrix = metrics.confusion_matrix(self.truth, self.pred_max)
      true_pos = [0] * len(classes)
      false_pos = [0] * len(classes)
      false_neg = [0] * len(classes)
      true_neg = [0] * len(classes)
      col_sum = np.sum(conf_matrix, axis=0)
      row_sum = np.sum(conf_matrix, axis=1)
      cum_sum = np.sum(conf_matrix)
      for k in range(0, len(classes)):
        true_pos[k] += conf_matrix[k, k]
        false_pos[k] += col_sum[k] - true_pos[k]
        false_neg[k] += row_sum[k] - true_pos[k]
        true_neg[k] += cum_sum - true_pos[k] - \
            false_pos[k] - false_neg[k]
    else:
      mlb = MultiLabelBinarizer()
      self.truth = mlb.fit_transform(self.truth)
      self.pred = mlb.transform(self.pred)
      classes = mlb.classes_
      self.truth = self.truth * 2
      np_sum = np.add(self.truth, self.pred)
      true_pos = [0] * len(classes)
      false_pos = [0] * len(classes)
      false_neg = [0] * len(classes)
      true_neg = [0] * len(classes)
      for i in range(0, len(classes)):
        true_pos[i] = np.sum(np_sum[:, i] == 3)
        true_neg[i] = np.sum(np_sum[:, i] == 0)
        false_pos[i] = np.sum(np_sum[:, i] == 1)
        false_neg[i] = np.sum(np_sum[:, i] == 2)

    # class wise evaluation
    for cls in classes:
      self.eval_result[cls] = {}
    for element in eval_list:
      if element in ['recall', 'true positive rate', 'sensitivity']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['recall'] = calc_recall(true_pos[i], false_neg[i])
      elif element in ['specificity', 'true negative rate']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['specificity'] = calc_specificity(true_neg[i], false_pos[i])
      elif element == 'accuracy':
        for i, cls in enumerate(classes):
          self.eval_result[cls]['accuracy'] = calc_acc(true_pos[i], true_neg[i], false_pos[i],
                                                       false_neg[i])
      elif element in ['f1_score', 'f1score', 'fscore']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['f1score'] = calc_f1score(true_pos[i], false_pos[i], false_neg[i])
      elif element in ['precision', 'positive predictive value', 'ppv']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['precision'] = calc_precision(true_pos[i], false_pos[i])
      elif element in ['negative predictive value', 'npv']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['npv'] = calc_npv(true_neg[i], false_neg[i])
      elif element in ['false negative rate', 'fnr']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['fnr'] = calc_fnr(false_neg[i], true_pos[i])
      elif element in ['false positive rate', 'fpr']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['fpr'] = calc_fpr(false_pos[i], true_neg[i])
      elif element in ['false discovery rate', 'fdr']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['fdr'] = calc_fdr(true_pos[i], false_pos[i])
      elif element in ['false omission rate', 'for']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['for'] = calc_for(false_neg[i], true_neg[i])
      elif element in ['matthews correlatin coefficient', 'mcc']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['mcc'] = calc_mcc(true_pos[i], true_neg[i], false_pos[i],
                                                  false_neg[i])
      elif element in ['kappa']:
        for i, cls in enumerate(classes):
          self.eval_result[cls]['kappa'] = calc_kappa(self.truth, self.pred_max)
      else:
        raise ValueError("invalid Evaluation Term")

    if plot_list:
      self.eval_plot_classification(plot_list, classes, true_pos, false_pos, false_neg, true_neg)
    if over_all:
      self.calc_overall()
    return self.eval_result, self.eval_plots

  def eval_regression(self, truth_file, pred_files, eval_list, plot_list, ensemble_weights=None):
    """
      This function calculates the evaluation matrices
      required by the user for regression problems.
    Args:
      truth_file: A csv file containing ids and truth annotations.
      pred_files: A list of csv files containing ids and prediction annotations.
      eval_list: A list of evaluation measures.
      plot_list: A list of evaluation plots.
      ensemble_weights: Weights for each class in case of ensemble, default None.
    Returns:
      A dictionary containing evaluation result.
    Raises: Invalid evaluation term if a term is not present in supported list.
    """
    self.eval_result = {}
    self.eval_plots = []
    self.read_data(truth_file, pred_files)
    eval_list = [element.strip().lower() for element in eval_list]
    if self.ensemble:
      ensemble_voting = "soft"
      self.eval_ensemble(ensemble_voting, ensemble_weights)
    for element in eval_list:
      if element in ['mae', 'mean absolute error']:
        self.eval_result['mae'] = calc_mae(self.truth, self.pred)
      elif element in ['mse', 'mean squared error']:
        self.eval_result['mse'] = calc_mse(self.truth, self.pred)
      elif element in ['rmse', 'root mean squared error']:
        self.eval_result['rmse'] = calc_rmse(self.truth, self.pred)
      else:
        raise ValueError("invalid Evaluation Term")
    if plot_list:
      self.eval_plot_regression(plot_list)
    return self.eval_result, self.eval_plots

  def eval_plot_classification(self, plot_list, classes, true_pos, false_pos, false_neg, true_neg):
    """
      This function plots the evaluation plots required
      by user for classification problems, like roc curve or confusion matrix.
    Args:
      plot_list: list of plots.
      classes: Array of class names
      true_pos: Number of true positives
      false_pos: Number of false positives
      false_neg: Number of false negatives
      true_neg: Number of true negatives
    Returns:
      None
    """

    plot_list = [element.strip().lower() for element in plot_list]
    for plot in plot_list:
      if plot in ['roc', 'receiver operating characteristics']:
        for i, cls in enumerate(classes):
          actual = [1 if t == i else 0 for t in self.truth]
          predicted = [p[i] for p in self.pred]
          fpr, tpr, _ = metrics.roc_curve(actual, predicted)
          fig = figure(title="ROC Curve " + str(cls), x_axis_label='FPR', y_axis_label='TPR')
          fig.line(fpr, tpr, line_width=2)
          self.eval_plots.append(fig)
      if plot in ["confusion matrix", "conf matrix"]:
        if self.multilabel:
          for i, _ in enumerate(classes):
            conf_mat = np.array([[true_pos[i], false_neg[i]], [false_pos[i], true_neg[i]]])
            cnf = plot_conf_mat(conf_mat, [classes[i], "~" + str(classes[i])])
        else:
          conf_mat = metrics.confusion_matrix(self.truth, self.pred_max)
          cnf = plot_conf_mat(conf_mat, classes)
        self.eval_plots.append(cnf)

  def eval_plot_regression(self, plot_list):
    """
      This function plots the evaluation plots required
      by user for regression problems, like residual plots or error bars.
    Args:
      plot_list: list of plots.
    Returns:
      None
    """

    plot_list = [element.strip().lower() for element in plot_list]
    for plot in plot_list:
      if plot in ["residual plot"]:
        fig = figure(plot_width=800, plot_height=500)
        fig.circle(self.pred, self.pred - self.truth, size=10, color="navy", alpha=0.5)
        self.eval_plots.append(fig)
      if plot in ["error bar"]:
        groups = ['Truth', 'Predicted']
        mean_vals = [np.mean(self.truth), np.mean(self.pred)]
        std_dev = [np.std(self.truth), np.std(self.pred)]
        top = [x + e for x, e in zip(mean_vals, std_dev)]
        down = [x - e for x, e in zip(mean_vals, std_dev)]
        y_range = (min(down) - 1, max(top) + 1)
        source = ColumnDataSource(data=dict(groups=groups, counts=mean_vals, upper=top, lower=down))
        fig = figure(
            x_range=groups,
            plot_height=400,
            toolbar_location=None,
            title="Error Bar(SD)",
            y_range=y_range)
        fig.vbar(
            x='groups',
            top='counts',
            width=0.9,
            source=source,
            legend="groups",
            line_color='white',
            fill_color=factor_cmap('groups', palette=["#972881", "#186f97"], factors=groups))
        fig.add_layout(
            Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay"))
        fig.xgrid.grid_line_color = None
        fig.legend.orientation = "horizontal"
        fig.legend.location = "top_center"
        self.eval_plots.append(fig)
        # Error bar with 95% confidence interval
        total_inst = len(self.truth)
        conf_inter = [2 * (sdev / math.sqrt(total_inst)) for sdev in std_dev]
        ci_top = [x + e for x, e in zip(mean_vals, conf_inter)]
        ci_down = [x - e for x, e in zip(mean_vals, conf_inter)]
        ci_y_range = (min(ci_down) - 1, max(ci_top) + 1)
        source = ColumnDataSource(
            data=dict(groups=groups, counts=mean_vals, upper=ci_top, lower=ci_down))
        fig = figure(
            x_range=groups,
            plot_height=400,
            toolbar_location=None,
            title="Error Bar(95% CI)",
            y_range=ci_y_range)
        fig.vbar(
            x='groups',
            top='counts',
            width=0.9,
            source=source,
            legend="groups",
            line_color='white',
            fill_color=factor_cmap('groups', palette=["#972881", "#186f97"], factors=groups))
        fig.add_layout(
            Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay"))
        fig.xgrid.grid_line_color = None
        fig.legend.orientation = "horizontal"
        fig.legend.location = "top_center"
        self.eval_plots.append(fig)

  def eval_ensemble(self, ensemble_voting, ensemble_weights):
    """
    This function will take multiple predictions from multiple models
    and will convert it into one single prediction output
    according to the voting strategy(hard or soft)
    hard voting will be used in case of classification and
    soft voting can be used for both classification(with probablities) and regression problems.
    Args:
      ensemble_voting: Type of strategy to be used for ensemble
      ensemble_weights: Weights array for classes to be used for ensemble
    Returns:
      None
    """
    if ensemble_voting == "hard":
      final_pred = np.array([])
      counts = {}
      for i in range(0, len(self.pred[0])):
        p_list = [int(np.argmax(p[i])) for p in self.pred]
        counts = Counter(p_list)
        p_list = sorted(p_list, key=lambda x: -counts[x])
        final_pred = np.append(final_pred, p_list[0])
      self.pred_max = final_pred
    else:
      final_pred = np.zeros_like(self.pred[0])
      if not ensemble_weights:
        ensemble_weights = [1] * len(self.pred)
      for i in range(0, len(self.pred)):
        final_pred = np.add(final_pred, self.pred[i] * ensemble_weights[i])
      final_pred = final_pred / float(np.sum(ensemble_weights))
      if len(final_pred.shape) > 1:
        self.pred_max = np.argmax(final_pred, axis=1)
    self.pred = final_pred

  def calc_overall(self):
    """
      function to calculate over all evaluation result.
    Args:
      None
    Returns:
      None
    """
    overall = defaultdict(int)
    for _, evl in self.eval_result.items():
      for el_key, el_val in evl.items():
        overall[el_key] += el_val

    overall = {k: round(v / len(self.eval_result.keys()), 3) for k, v in overall.items()}
    self.eval_result = {}
    self.eval_result["overall"] = overall
