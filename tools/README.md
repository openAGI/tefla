```Shell
python tools/save_graph.py --input_model models/model.py --output_model input_graph.pb

```

```Shell
python tools/freeze_graph.py --input_graph input_graph.pb --input_checkpoint /path/to/model-epoch-xxx.ckpt --output_graph output_graph.pb --output_node_names=predictions/Softmax

```
```Shell
python tools/load_frozen_graph.py --frozen_model output_graph.pb

```
```Shell
python predictv2.py --frozen_model output_graph.pb --training_cnf models/multiclass_cnf.py --predict_dir /path/to/data_dir --dataset_name name_fo_the_dataset

```

# Tool to format data for tefla
   - inputdata folder will have subfolders named as per class names; FOr example, if you have two class person and bike, inputdata will have two subfolders named as person and bike
```Shell
python data_process2.py --class_names person,bike --class_labels 0,1 --data_dir /path/to/inputdata
```
