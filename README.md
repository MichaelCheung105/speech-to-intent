# speech-to-intent

1. pip install requirements.txt
2. set PYTHONPATH={repository}/{src}
3. set ENVIRONMENT_VARIABLE={repository}

LSTM for time-series
CNN for spatial relationship

Result
64B = 30%
128B = 43%
256B = 55%
512B = 74%
1024B = 79%

512B + 16C = 83%, 77%
512B + 32C = 74%, 65%
512B + 64C = 55%, 50%

