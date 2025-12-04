step1:data_preprocess.ipynb 数据清洗，结果保存为raw_data\P1\P1_study2_cleaned.csv，合并每个实验的数据到data_pro1
step2:data_slice.ipynb 数据切片，保存到new_seg_data sliced_data
step3:sliding_window.ipynb 滑动窗口+特征计算，保存为feature.csv
step4:machine_learning.py 机器学习 machine_learning_autogluon.py autogluon机器学习