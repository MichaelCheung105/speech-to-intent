# Speech-to-Intent

### Prerequisite
To run the model training and evaluation pipeline, the following prerequisite is required. 
1. Install the required package using the "requirements.txt" file in the repository by **pip install requirements.txt**
2. Add the 'src' directory in the repository to PYTHONPATH by **set PYTHONPATH={your_repository}\src**
3. Add the environment variable "FLUENT_HOME"={your_repository}" by **set FLUENT_HOME={your_repository}\src**

### Model Training & Evaluation
A main.py module allows the use to train and evaluate the model.
The 'env_config.ini' file is used to adjust the config setting for model training and evaluation.

1. To train the model, go to the "DEFAULT" section of 'env_config.ini' and set the 'experiment_mode' to 'train'.
By default, the 'env_config.ini' file is in train mode already.
The other hyper-parameters are the settings that leads to the best performing model. 

2. To evaluate the model, go to the "DEFAULT" section of 'env_config_ini' and set the 'experiment_mode' to 'evaluate'.
Also, set the 'model_id' to the model you desired. The model IDs can be found in the './result/models' directory.
By default, the 'model_id' is set to be the best performing model.

### Report
A report can be found in the 'docs' directory named 'Report'.
Both jupyter notebook and html format is supported.
It outlined the model architecture, training method, performance of the model and findings.