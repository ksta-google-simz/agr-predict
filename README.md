
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

# AGR Prediction w/o label
You can now perform AGR (Age, Gender, Race) prediction on datasets without labels. This feature leverages the DeepFace library to analyze differences in age, gender, and race between original and anonymized images.
## Usage
1. Prepare your original and anonymized images following this structure:
```bash
.
├── celeb
│   ├── original
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── anon_10_102
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
```
- `original folder`: Contains the original images.

- `anon_{param} folder`: Contains images anonymized with a specific hyperparameter setting (num_inference_steps, anonymization_degree).

Run the following script to perform AGR predictions without labels:
```bash
python compare_agr_no_label.py
```
Ensure that you adjust the paths of `original_dir` and `anonymized_dir` in the script according to your dataset locations.

## Results

Upon executing the script, you will see the following results in the terminal:

- Total number of images analyzed

- Average absolute age difference, and proportions of age increase or decrease

- Proportion of maintained gender and race

- Detailed statistics of changes in gender and race (if applicable)

This functionality allows accurate evaluation of anonymization performance even when labels are not provided.