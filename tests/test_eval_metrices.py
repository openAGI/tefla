import tempfile
import random
import csv
import pytest
import numpy as np
import tensorflow as tf
from tefla.core.eval_metrices import Evaluation


class TestEvalMetrices(tf.test.TestCase):
  """
    Tests for Eval metrices
  """
  tmpdir = tempfile.TemporaryDirectory(dir="")
  tmpdirname = tmpdir.name
  @pytest.fixture(autouse=True)
  def data_gen(self):
    generate_datafiles(self.tmpdirname)

  def test_metrices(self):
    with self.test_session():
      evaluation_list = [
        'accuracy',
        'recall',
        'precision',
        'fpr',
        'for',
        'fnr',
        'mcc',
        'fdr',
        'specificity',
        'npv',
        'f1score']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_classification(self.tmpdirname +
                      '/truth_binary.csv', [self.tmpdirname +
                                '/pred_binary.csv'], evaluation_list, plot_list)
      target = {'pos': {'accuracy': 0.5,
              'recall': 0.2,
              'precision': 0.5,
              'fpr': 0.2,
              'for': 0.5,
              'fnr': 0.8,
              'fdr': 0.5,
              'mcc': 0.0,
              'specificity': 0.8,
              'npv': 0.5,
              'f1score': 0.286}}
      self.assertDictEqual(res['pos'], target['pos'])

  def test_multi(self):
    with self.test_session():
      evaluation_list = ['accuracy', 'recall']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_classification(self.tmpdirname +
                      '/truth_multi.csv', [self.tmpdirname +
                                 '/pred_multi.csv'], evaluation_list, plot_list)
      target = {
        'norm': {'accuracy': 0.72, 'recall': 0.2},
        'abnorm1': {'accuracy': 0.48, 'recall': 0.0},
        'abnorm2': {'accuracy': 0.6, 'recall': 0.2},
        'abnorm3': {'accuracy': 0.68, 'recall': 0.0},
        'abnorm4': {'accuracy': 0.68, 'recall': 0.0}}
      self.assertDictEqual(res, target)
      res_2, _ = evl.eval_classification(
        self.tmpdirname +
        '/truth_multi.csv',
        [
          self.tmpdirname +
          '/pred_multi.csv'],
        evaluation_list,
        plot_list,
        over_all=True)
      target_2 = {'overall': {'accuracy': 0.632, 'recall': 0.08}}
      self.assertDictEqual(res_2, target_2)

  def test_multi_convert_binary(self):
    with self.test_session():
      evaluation_list = ['accuracy', 'recall']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_classification(self.tmpdirname +
                      '/truth_multi.csv', [self.tmpdirname +
                                 '/pred_multi.csv'], evaluation_list, plot_list,convert_binary=True,binary_threshold=0.3)
      target = {
      'norm': {'accuracy': 0.72, 'recall': 0.2},
      '!norm': {'accuracy': 0.72, 'recall': 0.85}}
      self.assertDictEqual(res, target)

  def test_multi_withlabels(self):
    with self.test_session():
      evaluation_list = ['accuracy', 'recall']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_classification(
        self.tmpdirname +
        '/truth_multi_withlabel.csv',
        [
          self.tmpdirname +
          '/pred_multi_withlabel.csv'],
        evaluation_list,
        plot_list,
        class_names=[
          'A',
          'B',
          'C',
          'D',
          'E'])
      target = {
        'A': {'accuracy': 0.68, 'recall': 0.2},
        'B': {'accuracy': 0.6, 'recall': 0.4},
        'C': {'accuracy': 0.72, 'recall': 0.0},
        'D': {'accuracy': 0.72, 'recall': 0.0},
        'E': {'accuracy': 0.68, 'recall': 0.4}}
      self.assertDictEqual(res, target)

  def test_multi_ensemble(self):
    with self.test_session():
      evaluation_list = ['accuracy', 'recall']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_classification(
        self.tmpdirname +
        '/truth_multi_ens.csv',
        [
          self.tmpdirname +
          '/pred_multi_ens.csv',
          self.tmpdirname +
          '/pred_multi_ens_2.csv',
          self.tmpdirname +
          '/pred_multi_ens_3.csv',
          self.tmpdirname +
          '/pred_multi_ens_4.csv'],
        evaluation_list,
        plot_list,
        ensemble_voting="hard")
      target = {
        'norm': {'accuracy': 0.72, 'recall': 0.2},
        'abnorm1': {'accuracy': 0.6, 'recall': 0.2},
        'abnorm2': {'accuracy': 0.68, 'recall': 0.0},
        'abnorm3': {'accuracy': 0.64, 'recall': 0.2},
        'abnorm4': {'accuracy': 0.68, 'recall': 0.2}}
      self.assertDictEqual(target, res)
      res_2, _ = evl.eval_classification(
        self.tmpdirname +
        '/truth_multi_ens.csv',
        [
          self.tmpdirname +
          '/pred_multi_ens.csv',
          self.tmpdirname +
          '/pred_multi_ens_2.csv',
          self.tmpdirname +
          '/pred_multi_ens_3.csv',
          self.tmpdirname +
          '/pred_multi_ens_4.csv'],
        evaluation_list,
        plot_list,
        ensemble_voting="soft")
      target_2 = {
        'norm': {'accuracy': 0.72, 'recall': 0.0},
        'abnorm1': {'accuracy': 0.64, 'recall': 0.2},
        'abnorm2': {'accuracy': 0.56, 'recall': 0.2},
        'abnorm3': {'accuracy': 0.68, 'recall': 0.2},
        'abnorm4': {'accuracy': 0.64, 'recall': 0.0}}
      self.assertDictEqual(target_2, res_2)

  def test_multilabel(self):
    with self.test_session():
      evaluation_list = ['accuracy', 'recall', 'precision']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_classification(self.tmpdirname +
                      '/truth_multilabel.csv', [self.tmpdirname +
                                  '/pred_multilabel.csv'], evaluation_list, plot_list)
      target = {
        'A': {'accuracy': 0.786, 'recall': 0.667, 'precision': 1.0},
        'B': {'accuracy': 0.857, 'recall': 1.0, 'precision': 0.818},
        'C': {'accuracy': 0.714, 'recall': 0.8, 'precision': 0.8}}
      self.assertDictEqual(target, res)

  def test_reg_1(self):
    with self.test_session():
      evaluation_list = ['mae', 'rmse', 'mse']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_regression(self.tmpdirname +
                    '/truth_reg.csv', [self.tmpdirname +
                             '/truth_reg.csv'], evaluation_list, plot_list)
      target = {'mae': 0.0, 'rmse': 0.0, 'mse': 0.0}
      self.assertDictEqual(target, res)

  def test_reg_2(self):
    with self.test_session():
      evaluation_list = ['mae', 'rmse', 'mse']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_regression(self.tmpdirname +
                    '/truth_reg.csv', [self.tmpdirname +
                             '/pred_reg_1.csv'], evaluation_list, plot_list)
      target = {'mae': 0.539, 'rmse': 0.619, 'mse': 0.383}
      self.assertDictEqual(target, res)

  def test_reg_ensemble(self):
    with self.test_session():
      evaluation_list = ['mae', 'rmse', 'mse']
      plot_list = []
      evl = Evaluation()
      res, _ = evl.eval_regression(self.tmpdirname +
                    '/truth_reg.csv', [self.tmpdirname +
                             '/pred_reg_1.csv', self.tmpdirname +
                             '/pred_reg_2.csv', self.tmpdirname +
                             '/pred_reg_3.csv'], evaluation_list, plot_list)
      target = {'mae': 0.53, 'rmse': 0.554, 'mse': 0.307}
      self.assertDictEqual(target, res)


def generate_datafiles(dir_name):
  """
    Function to generate temp data files which will be used for test cases.
  """
  random.seed(30)
  np.random.seed(40)
  instance_list = ["inst_" + str(i) for i in range(0, 10)]
  truth_val = [0] * 5 + [1] * 5
  pred_val = [
    np.random.dirichlet(
      np.ones(2),
      size=1)[0].tolist() for _ in range(0, len(truth_val))]
  with open(dir_name + '/truth_binary.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "truth_val"])
    for value in zip(instance_list, truth_val):
      writer.writerow(value)

  with open(dir_name + '/pred_binary.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "neg", "pos"])
    for value in zip(instance_list, pred_val):
      writer.writerow((value[0],*value[1]))

  # for multiclass classification

  instance_list = ["inst_" + str(i) for i in range(0, 25)]
  truth_val = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
  random.shuffle(truth_val)
  pred_val = [
    np.random.dirichlet(
      np.ones(5),
      size=1)[0].tolist() for _ in range(0, len(truth_val))]
  with open(dir_name + '/truth_multi.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "truth_val"])
    for value in zip(instance_list, truth_val):
      writer.writerow(value)

  with open(dir_name + '/pred_multi.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "norm", "abnorm1", "abnorm2", "abnorm3", "abnorm4"])
    for value in zip(instance_list, pred_val):
      writer.writerow((value[0],*value[1]))

  # multiclass classification ensemble

  instance_list = ["inst_" + str(i) for i in range(0, 25)]
  truth_val = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
  random.shuffle(truth_val)
  pred_val = [
    np.random.dirichlet(
      np.ones(5),
      size=1)[0].tolist() for _ in range(0, len(truth_val))]
  pred_val_2 = [
    np.random.dirichlet(
      np.ones(5),
      size=1)[0].tolist() for _ in range(0, len(truth_val))]
  pred_val_3 = [
    np.random.dirichlet(
      np.ones(5),
      size=1)[0].tolist() for _ in range(0, len(truth_val))]
  pred_val_4 = [
    np.random.dirichlet(
      np.ones(5),
      size=1)[0].tolist() for _ in range(0, len(truth_val))]
  with open(dir_name + '/truth_multi_ens.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "truth_val"])
    for value in zip(instance_list, truth_val):
      writer.writerow(value)

  with open(dir_name + '/pred_multi_ens.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "norm", "abnorm1", "abnorm2", "abnorm3", "abnorm4"])
    for value in zip(instance_list, pred_val):
      writer.writerow((value[0],*value[1]))
  with open(dir_name + '/pred_multi_ens_2.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "norm", "abnorm1", "abnorm2", "abnorm3", "abnorm4"])
    for value in zip(instance_list, pred_val_2):
      writer.writerow((value[0],*value[1]))
  with open(dir_name + '/pred_multi_ens_3.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "norm", "abnorm1", "abnorm2", "abnorm3", "abnorm4"])
    for value in zip(instance_list, pred_val_3):
      writer.writerow((value[0],*value[1]))
  with open(dir_name + '/pred_multi_ens_4.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "norm", "abnorm1", "abnorm2", "abnorm3", "abnorm4"])
    for value in zip(instance_list, pred_val_4):
      writer.writerow((value[0],*value[1]))

  # Multilabel

  truth = [
    ['A', 'B'],
    ['A', 'B'],
    ['B'],
    ['C'],
    ['A', 'B', 'C'],
    ['A', 'C'],
    ['A', 'B', 'C'],
    ['A', 'B', 'C'],
    ['A', 'C'],
    ['B'],
    ['A', 'B', 'C'],
    ['B', 'C'],
    ['A', 'C'],
    ['C']]
  pred = [
    ['A', 'B', 'C'],
    ['B'],
    ['B', 'C'],
    ['B', 'C'],
    ['A', 'B', 'C'],
    ['C'],
    ['A', 'B'],
    ['B', 'C'],
    ['A', 'C'],
    ['B'],
    ['A', 'B', 'C'],
    ['B', 'C'],
    ['A', 'C'], ['B']]
  with open(dir_name + '/truth_multilabel.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "val"])
    for value in zip(instance_list, truth):
      writer.writerow(value)
  with open(dir_name + '/pred_multilabel.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "val"])
    for value in zip(instance_list, pred):
      writer.writerow(value)
  # Regression

  instance_list = ["inst_" + str(i) for i in range(0, 150)]
  truth_lables = [1] * 50 + [0] * 50
  random.shuffle(truth_lables)
  pred_lables_1 = [random.uniform(0, 1) for _ in range(0, len(truth_lables))]
  pred_labels_2 = [random.uniform(0, 1) for _ in range(0, len(truth_lables))]
  pred_labels_3 = [random.uniform(0, 1) for _ in range(0, len(truth_lables))]
  with open(dir_name + '/truth_reg.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "val"])
    for value in zip(instance_list, truth_lables):
      writer.writerow(value)
  with open(dir_name + '/pred_reg_1.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "val"])
    for value in zip(instance_list, pred_lables_1):
      writer.writerow(value)
  with open(dir_name + '/pred_reg_2.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "val"])
    for value in zip(instance_list, pred_labels_2):
      writer.writerow(value)
  with open(dir_name + '/pred_reg_3.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "val"])
    for value in zip(instance_list, pred_labels_3):
      writer.writerow(value)
  # multiclass with labels

  instance_list = ["inst_" + str(i) for i in range(0, 25)]
  truth_val = ['A'] * 5 + ['B'] * 5 + ['C'] * 5 + ['D'] * 5 + ['E'] * 5
  random.shuffle(truth_val)
  pred_val = [
    np.random.dirichlet(
      np.ones(5),
      size=1)[0].tolist() for _ in range(0, len(truth_val))]
  with open(dir_name + '/truth_multi_withlabel.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "truth_val"])
    for value in zip(instance_list, truth_val):
      writer.writerow(value)

  with open(dir_name + '/pred_multi_withlabel.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "norm", "abnorm1", "abnorm2", "abnorm3", "abnorm4"])
    for value in zip(instance_list, pred_val):
      writer.writerow((value[0],*value[1]))


if __name__ == "__main__":
    tf.test.main()
