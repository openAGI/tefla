python tools/save_graph.py --input_model models/model.py --output_model input_graph.pb


python tools/freeze_graph.py --input_graph input_graph.pb --input_checkpoint /path/to/model-epoch-xxx.ckpt --output_graph output_graph.pb --output_node_names=predictions/Softmax

python tools/load_frozen_graph.py --frozen_model output_graph.pb

python multiclass_predictv2.py --frozen_model output_graph.pb --training_cnf models/multiclass_cnf.py --predict_dir /path/to/data_dir --dataset_name name_fo_the_dataset
