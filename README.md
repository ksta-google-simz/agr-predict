
# AGR prediction w/ deepface
To run `agr.py`, you need to set up your files in the following structure.
```bash
.
├── fairface_label_val.csv
├── original
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── ...
│   └── 200.jpg
└── anon
    ├── 10_075
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   ├── ...
    │   └── 200.jpg
    ├── 10_100
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   ├── ...
    │   └── 200.jpg
    ├── 10_110
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   ├── ...
    │   └── 200.jpg
    └── ...
```
You can download the `fairface_label_val.csv` file from [here](https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view).

The images stored in the `anon` folder are anonymized using [FAWN](https://github.com/ksta-google-simz/fawn).

The images stored in the `original` folder are selected from the validation set of the [FairFace](https://github.com/dchen236/FairFace) dataset, specifically 200 samples with `padding=0.25`.

If you have followed the above directory and file structure, you can run `agr.py`.

## Results
After successfully running `agr.py`, a `result` directory will be created, containing evaluation outputs for each anonymized dataset.  
The folder will include the following files:

- `all_experiments_metrics.json`: Aggregated metrics from all experiments.
- `comparison_results_10_075.csv`, `comparison_results_10_100.csv`, `comparison_results_10_110.csv`, ...: CSV files comparing results between original and anonymized images.
- `metrics_10_075.json`, `metrics_10_100.json`, `metrics_10_110.json`, ...: Individual metrics for each hyperparameter: num_inference_steps, anonymization_degree.

The directory structure will look like this:

```bash
result/
├── all_experiments_metrics.json
├── comparison_results
│   ├── 10_075.csv
│   ├── 10_096.csv
│   ├── 10_098.csv
│   └── ...
└── metrics
    ├── 10_075.json
    ├── 10_096.json
    ├── 10_098.json
    └── ...
```

## Inference Test
If you wanna try to anonymize a image, make a folder named `test_img` and save your own image in the folder.

Now then, with running `one_file.py`, you can find the result in the terminal.

(Please modify `img_path` according to the path of your image in `one_file.py` before run.)