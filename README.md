# ChemBoost

ChemBoost is a chemical-language based drug - target affinity prediction framework. The models in ChemBoost leverage distributed chemical word vectors both in protein and ligand representations and achieve state-of-the-art level prediction performance on both BDB and KIBA data sets. 

To run the experiments from scratch:

- Download the dataset not in the repository [from this link](https://cmpe.boun.edu.tr/~riza.ozcelik/chemboost/data/chemboost_data.zip)
- Then run the command below for the experiment you want to replicate:

`python run_experiments.py {dataset_name} {model_name} {save_name}`

where `{dataset_name}` is replaced with either KIBA or BDB and `{model_name}` expects the input to be in the models in the paper. You can set `{save_name}` any filename of your choice.
