from __future__ import division, print_function, absolute_import

import inspect
import os
import os.path
from inspect import getmembers, isfunction
import re
import ast

import tefla
from tefla.core import base
from tefla.core import initializers
from tefla.core import iter_ops
from tefla.core import layer_arg_ops
from tefla.core import layers
from tefla.core import metrics
from tefla.core import logger
from tefla.core import losses
from tefla.core import lr_policy
from tefla.core import prediction
from tefla.core import summary
from tefla.core import training
from tefla.core import learning
from tefla.core import learning_ss
from tefla.da import data
from tefla.da import iterator
from tefla.da import standardizer
from tefla.da import tta
from tefla.dataset import base
from tefla.dataset import dataflow
from tefla.dataset import decoder
from tefla.dataset import image_to_tfrecords
from tefla.dataset import reader
from tefla.utils import util

MODULES = [(activations, 'tefla.core.activations'),
           (Layers, 'tefla.core.layers'),
           (initializations, 'tefla.core.initializers'),
           (metrics, 'tefla.core.metrics'),
           (losses, 'tefla.core.losses'),
           (logger, 'tefla.core.logger'),
           (layer_args, 'tefla.core.layer_arg_ops'),
           (iterator_ops, 'tefla.core.iter_ops'),
           (learning_policy, 'tefla.core.lr_policy'),
           (summaries, 'tefla.core.summary'),
           (utils, 'tefla.utils.util'),
           (dataflow, 'tefla.dataset.dataflow'),
           (base_data, 'tefla.dataset.base'),
           (decoder, 'tefla.dataset.decoder'),
           (reader, 'tefla.dataset.reader'),
           (tfrecords, 'tefla.dataset.image_to_tfrecords'),
           (data_augmentation, 'tefla.da.data'),
           (standardizer, 'tefla.da.standardizer'),
           (iterator, 'tefla.da.iterator'),
           (predictions, 'tefla.core.predictions'),
           (trainer, 'tefla.core.training'),
           (trainer_multi_gpu, 'tefla.core.learning'),
           (trainer_unsupervised, 'tefla.core.learning_ss'),
           ]

KEYWORDS = ['Input', 'x', 'Output', 'Examples', 'Args',
            'Returns', 'Raises', 'References', 'Links']


def top_level_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))


def top_level_classes(body):
    return (f for f in body if isinstance(f, ast.ClassDef))


def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)


def format_func_doc(docstring, header):

    rev_docstring = ''

    if docstring:
        # Erase 2nd lines
        docstring = docstring.replace('\n' + '    ' * 3, '')
        docstring = docstring.replace('    ' * 2, '')
        name = docstring.split('\n')[0]
        docstring = docstring[len(name):]
        if name[-1] == '.':
            name = name[:-1]
        docstring = '\n\n' + header_style(header) + docstring
        docstring = "# " + name + docstring

        # format arguments
        for o in ['Args']:
            if docstring.find(o + ':') > -1:
                args = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                args = args.replace('    ', ' - ')
                args = re.sub(r' - ([A-Za-z0-9_]+):', r' - **\1**:', args)
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + args
                else:
                    rev_docstring += '\n\n' + args

        for o in ['Returns', 'References', 'Links']:
            if docstring.find(o + ':') > -1:
                desc = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                desc = desc.replace('\n-', '\n\n-')
                desc = desc.replace('    ', '')
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + desc
                else:
                    rev_docstring += '\n\n' + desc

        rev_docstring = rev_docstring.replace('    ', '')
        rev_docstring = rev_docstring.replace(']\n(http', '](http')
        for keyword in KEYWORDS:
            rev_docstring = rev_docstring.replace(keyword + ':', '<h3>'
                                                  + keyword + '</h3>\n\n')
    else:
        rev_docstring = ""
    return rev_docstring


def format_method_doc(docstring, header):

    rev_docstring = ''

    if docstring:
        docstring = docstring.replace('\n' + '    ' * 4, '')
        docstring = docstring.replace('\n' + '    ' * 3, '')
        docstring = docstring.replace('    ' * 2, '')
        name = docstring.split('\n')[0]
        docstring = docstring[len(name):]
        if name[-1] == '.':
            name = name[:-1]
        docstring = '\n\n' + method_header_style(header) + docstring
        #docstring = "\n\n <h3>" + name + "</h3>" + docstring

        # format arguments
        for o in ['Args']:
            if docstring.find(o + ':') > -1:
                args = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                args = args.replace('    ', ' - ')
                args = re.sub(r' - ([A-Za-z0-9_]+):', r' - **\1**:', args)
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + args
                else:
                    rev_docstring += '\n\n' + args

        for o in ['Returns', 'References', 'Links']:
            if docstring.find(o + ':') > -1:
                desc = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                desc = desc.replace('\n-', '\n\n-')
                desc = desc.replace('    ', '')
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + desc
                else:
                    rev_docstring += '\n\n' + desc

        rev_docstring = rev_docstring.replace('    ', '')
        rev_docstring = rev_docstring.replace(']\n(http', '](http')
        for keyword in KEYWORDS:
            rev_docstring = rev_docstring.replace(keyword + ':', '<h5>'
                                                  + keyword + '</h5>\n\n')
    else:
        rev_docstring = ""
    return rev_docstring


def classesinmodule(module):
    classes = []
    tree = parse_ast(os.path.abspath(module.__file__).replace('.pyc', '.py'))
    for c in top_level_classes(tree.body):
        classes.append(eval(module.__name__ + '.' + c.name))
    return classes


def functionsinmodule(module):
    fn = []
    tree = parse_ast(os.path.abspath(module.__file__).replace('.pyc', '.py'))
    for c in top_level_functions(tree.body):
        fn.append(eval(module.__name__ + '.' + c.name))
    return fn


def enlarge_span(str):
    return '<span style="font-size:115%">' + str + '</span>'


def header_style(header):
    name = header.split('(')[0]
    bold_name = '<span style="color:black;"><b>' + name + '</b></span>'
    header = header.replace('self, ', '').replace('(', ' (').replace(' ', '  ')
    header = header.replace(name, bold_name)
    # return '<span style="display: inline-block;margin: 6px 0;font-size: ' \
    #        '90%;line-height: 140%;background: #e7f2fa;color: #2980B9;' \
    #        'border-top: solid 3px #6ab0de;padding: 6px;position: relative;' \
    #        'font-weight:600">' + header + '</span>'
    return '<span class="extra_h1">' + header + '</span>'


def method_header_style(header):
    name = header.split('(')[0]
    bold_name = '<span style="color:black"><b>' + name + '</b></span>'
    header = header.replace('self, ', '').replace('(', ' (').replace(' ', '  ')
    header = header.replace(name, bold_name)
    return '<span class="extra_h2">' + header + '</span>'



print('Starting...')
classes_and_functions = set()


def get_func_doc(name, func):
    doc_source = ''
    if name in SKIP:
        return  ''
    if name[0] == '_':
        return ''
    if func in classes_and_functions:
        return ''
    classes_and_functions.add(func)
    header = name + inspect.formatargspec(*inspect.getargspec(func))
    docstring = format_func_doc(inspect.getdoc(func), module_name + '.' +
                                header)

    if docstring != '':
        doc_source += docstring
        doc_source += '\n\n ---------- \n\n'

    return doc_source


def get_method_doc(name, func):
    doc_source = ''
    if name in SKIP:
        return  ''
    if name[0] == '_':
        return ''
    if func in classes_and_functions:
        return ''
    classes_and_functions.add(func)
    header = name + inspect.formatargspec(*inspect.getargspec(func))
    docstring = format_method_doc(inspect.getdoc(func), header)

    if docstring != '':
        doc_source += '\n\n <span class="hr_large"></span> \n\n'
        doc_source += docstring

    return doc_source


def get_class_doc(c):
    doc_source = ''
    if c.__name__ in SKIP:
        return ''
    if c.__name__[0] == '_':
        return ''
    if c in classes_and_functions:
        return ''
    classes_and_functions.add(c)
    header = c.__name__ + inspect.formatargspec(*inspect.getargspec(
        c.__init__))
    docstring = format_func_doc(inspect.getdoc(c), module_name + '.' +
                                header)

    method_doc = ''
    if docstring != '':
        methods = inspect.getmembers(c, predicate=inspect.ismethod)
        if len(methods) > 0:
            method_doc += '\n\n<h2>Methods</h2>'
        for name, func in methods:
            method_doc += get_method_doc(name, func)
        if method_doc == '\n\n<h2>Methods</h2>':
            method_doc = ''
        doc_source += docstring + method_doc
        doc_source += '\n\n --------- \n\n'

    return doc_source

for module, module_name in MODULES:

    # Handle Classes
    md_source = ""
    for c in classesinmodule(module):
        md_source += get_class_doc(c)

    # Handle Functions
    for func in functionsinmodule(module):
        md_source += get_func_doc(func.__name__, func)

    # save module page.
    # Either insert content into existing page,
    # or create page otherwise
    path = 'docs/templates/' + module_name.replace('.', '/')[8:] + '.md'
    if False: #os.path.exists(path):
        template = open(path).read()
        assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                 ' but missing {{autogenerated}} tag.')
        md_source = template.replace('{{autogenerated}}', md_source)
        print('...inserting autogenerated content into template:', path)
    else:
        print('...creating new page with autogenerated content:', path)
    subdir = os.path.dirname(path)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    open(path, 'w').write(md_source)
