# BBB_uncertainty_project
This repository is a fork from https://github.com/tongxiaochu/BBB_uncertainty_project.

Code for "Blood–Brain Barrier Penetration Prediction Enhanced by Uncertainty Estimation"

### Overview
Blood–brain barrier is a pivotal factor to be considered in the process of central nervous system (CNS) drug development, and it is of great significance to rapidly explore the blood–brain barrier permeability (BBBp) of compounds in silico in early drug discovery process. Here, we focus on whether and how uncertainty estimation methods improve in silico BBBp models. We briefly surveyed the current state of in silico BBBp prediction and uncertainty estimation methods of deep learning models, and curated an independent dataset to determine the reliability of the state-of-the-art algorithms. The results exhibit that, despite the comparable performance on BBBp prediction between graph neural networks-based deep learning models and conventional physicochemical-based machine learning models, the GROVER-BBBp model shows greatly improvement when using uncertainty estimations.

### Environment setup
```bash
conda create -n BBB_uncertainty python==3.9
conda activate BBB_uncertainty
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge rdkit
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

### Citation
Please cite our paper if you find it helpful. Thank you!

Tong, X., Wang, D., Ding, X. et al. Blood–brain barrier penetration prediction enhanced by uncertainty estimation. *J Cheminform* **14**, 44 (2022). https://doi.org/10.1186/s13321-022-00619-2

```bibtext
@article{tong2022blood,
  title={Blood--brain barrier penetration prediction enhanced by uncertainty estimation},
  author={Tong, Xiaochu and Wang, Dingyan and Ding, Xiaoyu and Tan, Xiaoqin and Ren, Qun and Chen, Geng and Rong, Yu and Xu, Tingyang and Huang, Junzhou and Jiang, Hualiang and others},
  journal={Journal of Cheminformatics},
  volume={14},
  number={1},
  pages={1--15},
  year={2022},
  publisher={BioMed Central}
}
```