# BBB_uncertainty_project
This repository is a fork from https://github.com/tongxiaochu/BBB_uncertainty_project.

Code for "Blood–Brain Barrier Penetration Prediction Enhanced by Uncertainty Estimation"

### Environment setup
```bash
conda create -n BBB_uncertainty python==3.9
conda activate BBB_uncertainty
conda install -c conda-forge rdkit
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge tqdm
conda install -c conda-forge dill
conda install -c conda-forge seaborn
pip install descriptastorus
conda install -c conda-forge scikit-learn
conda install -c conda-forge requests
```

### Process test dataset and save features
```bash
python save_features.py --data_path ../dataset/Sdata-process-flow-step5-testdata.csv
                        --features_generator rdkit_2d_normalized
                        --save_path ../dataset/Sdata-process-flow-step5-testdata.npz
                        --sequential
```

### Estimate uncertainty for S-data by GROVER
```bash
python main.py GROVER --data_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv
                      --features_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.npz
                      --separate_test_path dataset/Sdata-process-flow-step5-testdata.csv
                      --separate_test_features_path dataset/Sdata-process-flow-step5-testdata.npz
                      --dataset_type classification
                      --no_features_scaling
                      --ensemble_size 5
                      --num_folds 1
                      --epochs 60
                      --dropout 0.5
                      --batch_size 64
                      --activation PReLU
                      --attn_hidden 16
                      --attn_out 4
                      --aug_rate 0
                      --depth 6
                      --dist_coff 0.3
                      --ffn_hidden_size 10
                      --ffn_num_layers 3
                      --final_lr 8e-05
                      --init_lr 0.00008
                      --max_lr 0.0008
                      --hidden_size 8
                      --num_attn_head 4
                      --weight_decay 1e-7
                      --pred_times 100
                      --gpu 0
                      --save_dir ./BBBp_results/GROVER
```

### Estimate uncertainty for S-data by AttentiveFP
```bash
python main.py AttentiveFP --data_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv
                           --separate_test_path dataset/Sdata-process-flow-step5-testdata.csv
                           --dataset_type classification
                           --ensemble_size 5
                           --pred_times 100
                           --save_dir ./BBBp_results/AttentiveFP
```

### Estimate uncertainty for S-data by RL/MLP
```bash
python main.py MLP --data_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv
                   --separate_test_path dataset/Sdata-process-flow-step5-testdata.csv
                   --feature_type PCP
                   --dataset_type classification
                   --ensemble_size 5
                   --save_dir "./BBBp_results/MLP(PCP)"
```

### Draw figures for uncertainty analysis plot 
```bash
python uncertainty_analysis_plot.py --model_type GROVER
                                    --save_dir ../BBBp_results/GROVER
```

### Model performance for all BBBp models
```bash
python model_performance_plot.py
```
