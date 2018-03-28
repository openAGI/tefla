"""Tests for MetaGraphDef Transform Tool."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.tools import graph_transforms

from tefla.core import meta_graph_editor


def _make_asset_file_def_any(node_name):
  asset_file_def = meta_graph_pb2.AssetFileDef()
  asset_file_def.tensor_info.name = node_name
  any_message = Any()
  any_message.Pack(asset_file_def)
  return any_message


class MetaGraphTransformTest(tf.test.TestCase):

  def test_meta_graph_editor(self):

    with tf.Graph().as_default():
      with tf.Session(''):
        a = tf.placeholder(tf.int64, [1], name='a')
        b = tf.placeholder(tf.int64, [1], name='b')
        c = tf.placeholder(tf.int64, [1], name='c')
        _ = a * b
        _ = b * c
        base_meta_graph_def = tf.train.export_meta_graph()

    with tf.Graph().as_default():
      with tf.Session(''):
        a = tf.placeholder(tf.int64, [1], name='a')
        b = tf.placeholder(tf.int64, [1], name='b')
        _ = a * b
        meta_info_def = tf.MetaGraphDef.MetaInfoDef()
        meta_info_def.tags.append('tag_ab')

        expected_meta_graph_def = tf.train.export_meta_graph(meta_info_def=meta_info_def)
        # Graph rewriter clears versions field, so we expect that.
        expected_meta_graph_def.graph_def.ClearField('versions')
        # Graph rewriter adds an empty library field, so we expect that.
        expected_meta_graph_def.graph_def.library.CopyFrom(function_pb2.FunctionDefLibrary())

    input_names = ['a', 'b']
    output_names = ['mul:0']
    transforms = ['strip_unused_nodes']
    tags = ['tag_ab']
    print('AAAAAA: {}'.format(base_meta_graph_def))
    transformed_meta_graph_def = meta_graph_editor.meta_graph_editor(
        base_meta_graph_def, input_names, output_names, transforms, tags)

    self.assertEqual(expected_meta_graph_def, transformed_meta_graph_def)

  def test_get_shared_init_op(self):
    main_op = 'main_op'
    legacy_op = 'legacy_op'

    legacy_only = {tf.saved_model.constants.LEGACY_INIT_OP_KEY: [legacy_op]}
    main_and_legacy = {
        tf.saved_model.constants.MAIN_OP_KEY: [main_op],
        tf.saved_model.constants.LEGACY_INIT_OP_KEY: [legacy_op]
    }
    self.assertEqual(meta_graph_editor._get_shared_init_op({}), None)
    self.assertEqual(meta_graph_editor._get_shared_init_op(main_and_legacy), main_op)
    self.assertEqual(meta_graph_editor._get_shared_init_op(legacy_only), legacy_op)

  @tf.test.mock.patch.object(graph_transforms, 'TransformGraph')
  def test_gtt_transforms(self, graph_transform_mock):
    graph_def = tf.GraphDef()
    graph_def.node.extend([tf.NodeDef(name='z1', op='NoOp')])
    input_names = ['i1', 'i2']
    output_names = ['o1', 'o2']
    init_nodes = ['init1', 'init2']
    initializer_names = {'init': init_nodes}
    transforms = ['t1', 't2']

    expected_graph = tf.GraphDef()
    expected_graph.node.extend([tf.NodeDef(name='n1', op='NoOp')])
    graph_transform_mock.return_value = expected_graph
    transformed_graph_def = (meta_graph_editor._gtt_transforms(graph_def, input_names, output_names,
                                                               initializer_names, transforms))

    self.assertEqual(transformed_graph_def, expected_graph)
    graph_transform_mock.assert_called_once_with(graph_def, input_names, output_names + init_nodes,
                                                 transforms)

  @tf.test.mock.patch.object(meta_graph_editor, '_freeze_graph_with_def_protos')
  def test_freeze_transform(self, freeze_mock):
    graph_def = tf.GraphDef()
    graph_def.node.extend([tf.NodeDef(name='z1', op='NoOp')])
    output_names = ['o1', 'o2']
    table_init_names = ['t1', 't2']
    main_op = 'main_op'
    legacy_op = 'legacy_op'
    initializer_names = {
        'foo_init': ['init1', 'init2'],
        tf.GraphKeys.TABLE_INITIALIZERS: table_init_names,
        tf.saved_model.constants.MAIN_OP_KEY: [main_op],
        tf.saved_model.constants.LEGACY_INIT_OP_KEY: [legacy_op]
    }
    expected_graph_def = tf.GraphDef()
    graph_def.node.extend([tf.NodeDef(name='n1', op='NoOp')])
    freeze_mock.return_value = expected_graph_def
    saver_def = tf.train.SaverDef()
    saver_def.filename_tensor_name = 'f1'
    checkpoint_path = '/checkpoint/path'
    transformed_graph_def, transformed_initializer_names = (meta_graph_editor._freeze_transform(
        graph_def, output_names, initializer_names, saver_def, checkpoint_path))
    self.assertEqual(transformed_graph_def, expected_graph_def)
    expected_initializer_names = {
        tf.GraphKeys.TABLE_INITIALIZERS: table_init_names,
        tf.saved_model.constants.MAIN_OP_KEY: [main_op],
        tf.saved_model.constants.LEGACY_INIT_OP_KEY: [legacy_op]
    }
    self.assertEqual(transformed_initializer_names, expected_initializer_names)
    freeze_mock.assert_called_once_with(graph_def, output_names, table_init_names, main_op,
                                        saver_def, checkpoint_path)

  def test_clean_save_and_restore(self):
    graph_def = tf.GraphDef()
    save_name = 'save_1/SaveV2'
    save_tensor_name = save_name + '/tensor_names'
    save_tensor_shape = save_name + '/shape_and_slices'
    save_op = graph_def.node.add()
    save_op.name = save_name
    save_op.op = 'NoOp'
    save_name_op = graph_def.node.add()
    save_name_op.name = save_tensor_name
    save_name_op.op = 'NoOp'
    save_shape_op = graph_def.node.add()
    save_shape_op.name = save_tensor_shape
    save_shape_op.op = 'NoOp'

    types = [types_pb2.DT_INT32, types_pb2.DT_FLOAT, types_pb2.DT_INT32]
    names = [tf.compat.as_bytes('/foo'), tf.compat.as_bytes('/bar'), tf.compat.as_bytes('/baz')]
    shapes = [
        tf.compat.as_bytes('100 10 0,100:0,10'),
        tf.compat.as_bytes('150 11 0,150:0,11'),
        tf.compat.as_bytes('101 12 0,101:0,12')
    ]

    expected_types = [types[0], types[2]]
    expected_names = [names[0], names[2]]
    expected_shapes = [shapes[0], shapes[2]]

    save_op.attr['dtypes'].list.type[:] = types
    save_name_op.attr['value'].tensor.string_val[:] = names
    save_name_op.attr['value'].tensor.tensor_shape.dim.add().size = len(names)
    save_name_op.attr['_output_shapes'].list.shape.add().dim.add().size = len(names)

    save_shape_op.attr['value'].tensor.string_val[:] = shapes
    save_shape_op.attr['value'].tensor.tensor_shape.dim.add().size = len(shapes)
    save_shape_op.attr['_output_shapes'].list.shape.add().dim.add().size = len(shapes)

    meta_graph_editor._clean_save_and_restore(graph_def, save_op, ['/bar'])
    self.assertEqual(save_op.attr['dtypes'].list.type[:], expected_types)
    self.assertEqual(save_name_op.attr['value'].tensor.string_val[:], expected_names)
    self.assertEqual(save_name_op.attr['value'].tensor.tensor_shape.dim[0].size,
                     len(expected_names))
    self.assertEqual(save_name_op.attr['_output_shapes'].list.shape[0].dim[0].size,
                     len(expected_names))

    self.assertEqual(save_shape_op.attr['value'].tensor.string_val[:], expected_shapes)
    self.assertEqual(save_shape_op.attr['value'].tensor.tensor_shape.dim[0].size,
                     len(expected_shapes))
    self.assertEqual(save_shape_op.attr['_output_shapes'].list.shape[0].dim[0].size,
                     len(expected_shapes))

  @tf.test.mock.patch.object(meta_graph_editor, '_clean_save_and_restore')
  @tf.test.mock.patch.object(meta_graph_editor, '_gtt_transforms')
  def test_sparsify_gather_transform(self, gtt_mock, clean_save_restore_mock):
    # Initial graph def.
    graph_def = tf.GraphDef()
    variable_op = graph_def.node.add()
    variable_op.name = '/foo/part_1'

    constant_op = graph_def.node.add()
    constant_op.name = '/bar'

    # Transformed graph def.
    transformed_graph_def = tf.GraphDef()
    constant_op = transformed_graph_def.node.add()
    constant_op.name = '/foo'

    sparsify_shared_init_op_name = 'sparify_gather_init_op'
    new_table_init_names = ['table1', 'table2']
    init_op = transformed_graph_def.node.add()
    init_op.name = sparsify_shared_init_op_name
    init_op.input.extend(['^' + f for f in new_table_init_names])

    saver_op = transformed_graph_def.node.add()
    saver_op.name = 'save_1/SaveV2'

    orig_table_init_names = ['orig_table_init_1', 'orig_table_init_2']

    legacy_op_name = 'legacy_op'
    legacy_op = transformed_graph_def.node.add()
    legacy_op.name = legacy_op_name
    legacy_op.input.extend(['^' + f for f in orig_table_init_names])

    input_names = ['i1', 'i2']
    output_names = ['o1', 'o2']

    initializer_names = {
        'foo_init': ['init1', 'init2'],
        tf.GraphKeys.TABLE_INITIALIZERS: orig_table_init_names,
        tf.saved_model.constants.LEGACY_INIT_OP_KEY: [legacy_op_name]
    }
    checkpoint_path = '/path/to/checkpoint'

    expected_initializer_names = {
        'foo_init': ['init1', 'init2'],
        tf.GraphKeys.TABLE_INITIALIZERS: (orig_table_init_names + new_table_init_names),
        tf.saved_model.constants.LEGACY_INIT_OP_KEY: [legacy_op_name]
    }

    expected_sparsify_cmd = [
        'sparsify_gather(input_checkpoint="%s", group_init_node="%s")' %
        (checkpoint_path, sparsify_shared_init_op_name)
    ]

    # Expected graph def.
    expected_graph_def = tf.GraphDef()
    constant_op = expected_graph_def.node.add()
    constant_op.name = '/foo'

    saver_op = expected_graph_def.node.add()
    saver_op.name = 'save_1/SaveV2'

    legacy_op_name = 'legacy_op'
    legacy_op = expected_graph_def.node.add()
    legacy_op.name = legacy_op_name
    legacy_op.input.extend(['^' + f for f in orig_table_init_names + new_table_init_names])

    gtt_mock.return_value = transformed_graph_def
    graph_def_result, init_names_result = (meta_graph_editor._sparsify_gather_transform(
        graph_def, input_names, output_names, initializer_names, checkpoint_path))

    gtt_mock.assert_called_once_with(graph_def, input_names, output_names, initializer_names,
                                     expected_sparsify_cmd)

    clean_save_restore_mock.assert_called_once_with(transformed_graph_def, saver_op,
                                                    ['/bar', '/foo'])

    self.assertEqual(expected_graph_def, graph_def_result)
    self.assertEqual(expected_initializer_names, init_names_result)

  @tf.test.mock.patch.object(meta_graph_editor, '_gtt_transforms')
  @tf.test.mock.patch.object(meta_graph_editor, '_freeze_transform')
  @tf.test.mock.patch.object(meta_graph_editor, '_sparsify_gather_transform')
  def test_do_transforms(self, sparsify_mock, freeze_mock, gtt_mock):
    graph_def = tf.GraphDef()
    constant_op = graph_def.node.add()
    constant_op.name = 'c1'

    input_names = ['i1', 'i2']
    output_names = ['o1', 'o2']
    initializer_names = {
        'foo_init': ['init1', 'init2'],
        tf.GraphKeys.TABLE_INITIALIZERS: ['table1'],
        tf.saved_model.constants.LEGACY_INIT_OP_KEY: ['legacy_op']
    }

    transforms = ['foo', 'freeze_graph', 'bar', 'sparsify_gather', 'baz']

    sparsify_mock.return_value = (graph_def, initializer_names)
    freeze_mock.return_value = (graph_def, initializer_names)
    gtt_mock.return_value = graph_def

    graph_def_result, initializer_names_result = (meta_graph_editor._do_transforms(
        graph_def, input_names, output_names, initializer_names, transforms))

    sparsify_mock.assert_called_once_with(graph_def, input_names, output_names, initializer_names,
                                          None)

    freeze_mock.assert_called_once_with(graph_def, output_names, initializer_names, None, None)

    gtt_mock.assert_has_calls([
        tf.test.mock.call(graph_def, input_names, output_names, initializer_names, ['foo']),
        tf.test.mock.call(graph_def, input_names, output_names, initializer_names, ['bar']),
        tf.test.mock.call(graph_def, input_names, output_names, initializer_names, ['baz'])
    ])
    self.assertEqual(graph_def_result, graph_def)
    self.assertEqual(initializer_names, initializer_names_result)

  def test_add_new_inits_to_collection(self):
    meta_graph_def = tf.MetaGraphDef()

    orig_table_inits = ['t1', 't2']
    new_table_inits = ['t3', 't4']

    meta_graph_def.collection_def[tf.GraphKeys.TABLE_INITIALIZERS].node_list.value.extend(
        orig_table_inits)
    updated_init_names = {tf.GraphKeys.TABLE_INITIALIZERS: orig_table_inits + new_table_inits}

    meta_graph_editor._add_new_inits_to_collection(meta_graph_def, updated_init_names)

    self.assertEqual(meta_graph_def.collection_def[tf.GraphKeys.TABLE_INITIALIZERS].node_list.value,
                     orig_table_inits + new_table_inits)

  @tf.test.mock.patch.object(graph_transforms, 'TransformGraph')
  @tf.test.mock.patch.object(meta_graph_editor, '_freeze_graph_with_def_protos')
  def test_freeze_then_sparsify(self, freeze_mock, graph_transform_mock):
    tag_name = 'tag'
    input_nodes = 'input_nodes'
    output_nodes = 'output_nodes'
    freeze_transform = 'freeze_graph'
    sparsify_transform = 'sparsify_gather'

    base_meta_graph_def = tf.MetaGraphDef()

    # Add a table initializer.
    table_init_name = 'table_init'
    node_def = tf.NodeDef(name=table_init_name, op='InitializeTableV2')
    base_meta_graph_def.graph_def.node.extend([node_def])

    # Add a group_deps node.
    group_deps_name = 'group_deps'
    node_def = tf.NodeDef(name=group_deps_name, op='NoOp')
    node_def.input.extend(['^table_init'])
    base_meta_graph_def.graph_def.node.extend([node_def])

    base_meta_graph_def.collection_def[tf.GraphKeys.TABLE_INITIALIZERS].node_list.value.extend(
        [table_init_name])
    base_meta_graph_def.collection_def[
        tf.saved_model.constants.LEGACY_INIT_OP_KEY].node_list.value.extend([group_deps_name])

    # Expected metagraphdef.
    expected_meta_graph_def = tf.MetaGraphDef()
    expected_meta_graph_def.CopyFrom(base_meta_graph_def)
    expected_meta_graph_def.meta_info_def.tags.append(tag_name)

    transformed_graph_def = tf.GraphDef()
    transformed_graph_def.CopyFrom(expected_meta_graph_def.graph_def)
    freeze_mock.return_value = transformed_graph_def
    graph_transform_mock.return_value = transformed_graph_def

    # Add unsaved init node.
    unsaved_init_name = 'unsaved_node'
    node_def = tf.NodeDef(name=unsaved_init_name, op='NoOp')
    base_meta_graph_def.graph_def.node.extend([node_def])

    # Add a saver.
    base_meta_graph_def.saver_def.filename_tensor_name = 'node1'
    base_meta_graph_def.saver_def.save_tensor_name = 'node3'
    base_meta_graph_def.saver_def.restore_op_name = 'node6'

    transformed_meta_graph_def = meta_graph_editor.meta_graph_editor(
        base_meta_graph_def, [input_nodes], [output_nodes], [freeze_transform, sparsify_transform],
        [tag_name])

    self.assertEqual(expected_meta_graph_def, transformed_meta_graph_def)
    freeze_mock.assert_called_once_with(base_meta_graph_def.graph_def, [output_nodes],
                                        [table_init_name], group_deps_name,
                                        base_meta_graph_def.saver_def, None)
    graph_transform_mock.assert_called_once_with(
        transformed_graph_def, [input_nodes], [output_nodes, group_deps_name, table_init_name],
        [sparsify_transform + '(group_init_node="sparify_gather_init_op")'])

  def test_connect_to_shared_init_op(self):
    group_deps_name = 'group_deps'
    init_node_1 = 'table_init_1'
    init_node_2 = 'table_init_2'

    orig_graph_def = tf.GraphDef()
    expected_graph_def_1 = tf.GraphDef()

    meta_graph_editor._connect_to_shared_init_op(orig_graph_def, group_deps_name, [])
    self.assertEqual(expected_graph_def_1, orig_graph_def)

    expected_graph_def_2 = tf.GraphDef()
    node_def = tf.NodeDef(name=group_deps_name, op='NoOp')
    node_def.input.extend(['^' + init_node_1, '^' + init_node_2])
    expected_graph_def_2.node.extend([node_def])

    meta_graph_editor._connect_to_shared_init_op(orig_graph_def, group_deps_name,
                                                 [init_node_1, init_node_2])
    self.assertEqual(expected_graph_def_2, orig_graph_def)

  def test_add_pruned_collection_node(self):
    # Note: This also tests _is_removed().
    collection_name = 'node_collection'
    base_meta_graph_def = tf.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].node_list.value.extend(
        ['node1', 'node2', 'node3', 'node4', '/a/a_1', '/b/b_1'])

    meta_graph_def = tf.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5', '/a', '/b/b_1']
    meta_graph_editor._add_pruned_collection(base_meta_graph_def, meta_graph_def, collection_name,
                                             removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_nodes = ['node1', 'node3', '/a/a_1']
    self.assertEqual(expected_nodes, collection.node_list.value)

  def test_add_pruned_collection_int(self):
    collection_name = 'int_collection'
    base_meta_graph_def = tf.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].int64_list.value[:] = ([10, 20, 30, 40])

    meta_graph_def = tf.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5', '/a', '/b/b_1']
    meta_graph_editor._add_pruned_collection(base_meta_graph_def, meta_graph_def, collection_name,
                                             removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_ints = [10, 20, 30, 40]
    self.assertEqual(expected_ints, collection.int64_list.value)

  def test_add_pruned_collection_proto_in_any_list(self):
    # Note: This also tests _is_removed_mentioned().
    collection_name = 'proto_collection'
    base_meta_graph_def = tf.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].any_list.value.extend([
        _make_asset_file_def_any('node1'),
        _make_asset_file_def_any('node2'),
        _make_asset_file_def_any('node3'),
        _make_asset_file_def_any('node4'),
        _make_asset_file_def_any('/a/a_1'),
        _make_asset_file_def_any('/b/b_1')
    ])

    meta_graph_def = tf.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5', '/a', '/b/b_1']
    meta_graph_editor._add_pruned_collection(base_meta_graph_def, meta_graph_def, collection_name,
                                             removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_protos = [
        _make_asset_file_def_any('node1'),
        _make_asset_file_def_any('node3'),
        _make_asset_file_def_any('/a/a_1'),
    ]
    self.assertEqual(expected_protos, collection.any_list.value[:])

  def test_add_pruned_collection_proto_in_bytes_list(self):
    # Note: This also tests _is_removed_mentioned().
    collection_name = 'proto_collection'
    base_meta_graph_def = tf.MetaGraphDef()
    base_meta_graph_def.collection_def[collection_name].bytes_list.value.extend([
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('node1'))),
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('node2'))),
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('node3'))),
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('node4'))),
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('/a/a_1'))),
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('/b/b_1')))
    ])

    meta_graph_def = tf.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5', '/a', '/b/b_1']
    meta_graph_editor._add_pruned_collection(base_meta_graph_def, meta_graph_def, collection_name,
                                             removed_op_names)

    collection = meta_graph_def.collection_def[collection_name]

    expected_values = [
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('node1'))),
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('node3'))),
        tf.compat.as_bytes(tf.compat.as_str_any(_make_asset_file_def_any('/a/a_1'))),
    ]
    self.assertEqual(expected_values, collection.bytes_list.value[:])

  def test_add_pruned_saver(self):
    base_meta_graph_def = tf.MetaGraphDef()

    base_meta_graph_def.saver_def.filename_tensor_name = 'node1'
    base_meta_graph_def.saver_def.save_tensor_name = 'node3'
    base_meta_graph_def.saver_def.restore_op_name = 'node6'

    meta_graph_def = tf.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_editor._add_pruned_saver(base_meta_graph_def, meta_graph_def, removed_op_names)

    # TODO(n3011): For now the saver is just copied unchanged
    self.assertEqual(base_meta_graph_def.saver_def, meta_graph_def.saver_def)

  def test_add_pruned_signature(self):
    base_meta_graph_def = tf.MetaGraphDef()

    signature_name_keep = 'test_signature_keep'
    base_sig_keep = base_meta_graph_def.signature_def[signature_name_keep]
    base_sig_keep.inputs['input_1'].name = 'input_1'
    base_sig_keep.outputs['output_1'].name = 'output_1'

    signature_name_remove = 'test_signature_remove'
    base_sig_remove = base_meta_graph_def.signature_def[signature_name_remove]
    base_sig_remove.inputs['node2'].name = 'node2'
    base_sig_remove.outputs['output_1'].name = 'output_1'

    meta_graph_def = tf.MetaGraphDef()
    removed_op_names = ['node2', 'node4', 'node5']
    meta_graph_editor._add_pruned_signature(base_meta_graph_def, meta_graph_def,
                                            signature_name_keep, removed_op_names)
    meta_graph_editor._add_pruned_signature(base_meta_graph_def, meta_graph_def,
                                            signature_name_remove, removed_op_names)

    self.assertTrue(signature_name_keep in meta_graph_def.signature_def)
    sig_keep = meta_graph_def.signature_def[signature_name_keep]
    self.assertEqual(base_sig_keep, sig_keep)

    self.assertFalse(signature_name_remove in meta_graph_def.signature_def)


if __name__ == '__main__':
  tf.test.main()
