# speech-to-intent

1. pip install requirements.txt
2. set PYTHONPATH={repository}/{src}
3. set ENVIRONMENT_VARIABLE={repository}

LSTM for time-series
CNN for spatial relationship

Result (32C)
64B = 30%
128B = 43%
256B = 55%
512B = 74%
1024B = 79%

512B + 16C = 83%, 77%
512B + 32C = 74%, 65%
512B + 64C = 55%, 50%

20210714003953 - 512B + 16C = 83%, 77%, ep65:0.577 val loss
(Should try adding padding (3,3) to the above)
20210717031231 - Adding noise to the above leads to 0.375 at epoch 153 (88%, 85%)
20210717031231 - Adding noise to the above leads to 0.285 at epoch 140 (91.6%, 87.7%)

20210717012343 - 512B + 16C(K33,D22,S33,P33) + 16C(K15) = 85%, 78%, ep20:0.496