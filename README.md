Code workflow explanation:

Step 1 – data_preprocess.ipynb
Performs raw data cleaning and timestamp alignment. Processed files are saved as raw_data/P1/P1_study2_cleaned.csv, and all modalities from each experiment are then merged into the data_pro1 directory.

Step 2 – data_slice.ipynb
Segments the synchronized signals into valid driving intervals and saves them into new_seg_data / sliced_data.

Step 3 – sliding_window.ipynb
Applies sliding-window processing and extracts multimodal features, saved as feature.csv.

Step 4 – machine_learning.py
Runs traditional machine-learning classifiers.
Step 4 – machine_learning_autogluon.py
Runs AutoML classification using AutoGluon.

Raw dataset explanation:

Pxx_study1.csv – Eye-tracking data (Experiment 1)

Pxx_study1.txt – Vehicle speed data (Experiment 1)

Pxx_study1_pedal.txt – Steering, throttle, and brake data (Experiment 1)

Pxx_study2.csv – Eye-tracking data (Experiment 2)

Pxx_study2.txt – Vehicle speed data (Experiment 2)

Pxx_study2_pedal.txt – Steering, throttle, and brake data (Experiment 2)