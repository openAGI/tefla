"""Wrapper to Evaluate model
"""
# -------------------------------------------------------------------#
# Copyright 2019 The Tefla Authors. All Rights Reserved.
# -------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import
import ast
import click
from prettytable import PrettyTable
from bokeh.io import save
from tefla.core.eval_metrices import Evaluation


class ListAsArguments(click.Option):
  """
    class to overload type_cast_value method from click.option
  """

  def type_cast_value(self, ctx, value):
    try:
      return ast.literal_eval(value)
    except BaseException:
      raise click.BadParameter(value)


# pylint: disable=no-value-for-parameter
@click.command()
@click.option(
    '--truth_file',
    default=None,
    show_default=True,
    required=True,
    help='Path to file containing ground truth')
@click.option(
    '--pred_files',
    default='[]',
    cls=ListAsArguments,
    show_default=True,
    required=True,
    help='Path to file containing  predictions.')
@click.option(
    '--eval_list',
    default='[]',
    cls=ListAsArguments,
    show_default=True,
    required=True,
    help='List of Evaluation matrices to be evaluated.')
@click.option(
    '--plot_list',
    default='[]',
    cls=ListAsArguments,
    show_default=True,
    help='List of Evaluation plots.')
@click.option(
    '--over_all',
    default=False,
    show_default=True,
    help='Flag If overall results are required instead of classwise results.')
@click.option(
    '--ensemble_voting',
    default="soft",
    show_default=True,
    help='The type of voting strategy to be used incase of ensemble.')
@click.option(
    '--ensemble_weights',
    default='[]',
    cls=ListAsArguments,
    show_default=True,
    help='Weights in case of ensemble with weights')
@click.option(
    '--class_names', default='[]', cls=ListAsArguments, show_default=True, help='Name of classes')
@click.option(
    '--convert_binary',
    default=False,
    show_default=True,
    help='Flag to indicate if problem should be evalauted as binary(normal vs abnormal)')
@click.option(
    '--binary_threshold',
    default=0.5,
    show_default=True,
    help='Threshold value to determine nomral and abnormal.')
@click.option(
    '--save_dir', default='.', show_default=True, help='Path where evaluation plots will be saved.')
@click.option(
    '--eval_type',
    default=None,
    show_default=True,
    required=True,
    help='Evaluation type classification or Regression.')
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def main(truth_file, pred_files, eval_list, plot_list, over_all, ensemble_voting, ensemble_weights,
         class_names, convert_binary, binary_threshold, save_dir, eval_type):
  """
    wrapper function to call eval_metrices api.
  """
  eval_model = Evaluation()
  if eval_type.lower() == 'classification':
    evl_result, evl_plots = eval_model.eval_classification(
        truth_file,
        pred_files,
        eval_list,
        plot_list,
        over_all=over_all,
        ensemble_voting=ensemble_voting,
        ensemble_weights=ensemble_weights,
        class_names=class_names,
        convert_binary=convert_binary,
        binary_threshold=binary_threshold)
  elif eval_type.lower() == 'regression':
    evl_result, evl_plots = eval_model.eval_regression(
        truth_file, pred_files, eval_list, plot_list, ensemble_weights=ensemble_weights)
  else:
    raise ValueError("invalid option, provide either classification or regression")

  p_tab = PrettyTable()
  p_tab.field_names = ["class"] + eval_list
  for keys, values in evl_result.items():
    p_tab.add_row([keys] + [values[evl] for evl in eval_list])
  print(p_tab)
  if evl_plots:
    for i, plt in enumerate(evl_plots):
      save(plt, filename=save_dir + '/' + str(i) + '.html')


if __name__ == '__main__':
  main()
