# Speech-to-Intent

### Introduction
This project is an interview assignment concerning speech recognition.
Since the data is proprietary, you can skip the **Prerequisite** section of this README.md file.
You can navigate the code by cloning the repository and following the **Model Training & Evaluation** section of this README.md file.
A **report** outlining the model architecture, training method, performance and findings can be found in the 'doc' directory.

### Prerequisite
To run the model training and evaluation pipeline, the following prerequisite is required: 
1. Place the *.npz data files in the 'data' directory
2. Install the required packages using the "requirements.txt" file by executing **pip install -r requirements.txt**
3. Add the 'src' directory to PYTHONPATH by executing **set PYTHONPATH={your_directory_address}\src**
4. Add the environment variable FLUENT_HOME as {your_directory_address} by executing **set FLUENT_HOME={your_directory_address}**

### Model Training & Evaluation
A main.py module allows the user to train and evaluate the model.
The 'env_config.ini' file is used to adjust the config setting for model training and evaluation.

1. To train the model, go to the "DEFAULT" section of 'env_config.ini' and set the 'experiment_mode' to 'train'.
By default, the 'env_config.ini' file is in train mode already.
The other hyper-parameters are the settings that lead to the best-performing model. 

2. To evaluate the model, go to the "DEFAULT" section of 'env_config_ini' and set the 'experiment_mode' to 'evaluate'.
Also, set the 'model_id' to the model you desired. The model IDs can be found in the './result/models' directory.
By default, the 'model_id' is set to be the best-performing model.

### Report
A report can be found in the 'docs' directory named 'Report'.
Both Jupiter notebook and the HTML format are supported.
It outlined the model architecture, training method, performance of the model, and findings.
