# Inefficient-AC-detection

This is the code repository for the paper _____________________, you can find the paper directly through **[here](https://www.google.com)**(to be updated)


## 1. Environment Setup
The experiment is conducted under: Windows 10 with Python 3.7/3.8 as the developing environment.

Use `pip install -r requirements.txt` to install all the packages. 

**In short, You must have `xgboost, numpy, seaborn, matplotlib, pandas, shap, warnings, scikit-learn` installed.**

## 2. Data Compilation
We provide the compiled version of our experiment data `data_compiled.csv`, it can also be accessed through Mendeley Data by **[here](https://data.mendeley.com/datasets/2932z9dz6g/3)**. 

The raw data before compilation will not be made available due to privacy reasons.

## 3. Training the XGBoost Model

Again, you must have all the packages above installed.

Run the `room_xgboost_training.py` to train the model, we use xgboost squared regressor and cross validation to do the training. Each room's model has been boosted for 300 rounds, and we used the [SMOTE](https://doi.org/10.1613/jair.953) algorithm to help with the imbalance distribution of the data.

Here is a simple demonstration of the data distribution before the [SMOTE](https://doi.org/10.1613/jair.953) algorithm.
![SMOTE_before](SMOTE_before.png)

After the [SMOTE](https://doi.org/10.1613/jair.953) algorithm, the distribution for AC below or above 0.7 is balanced.
![SMOTE_after](SMOTE_After.png)
 
The training shall take about two hours with the `data_compiled` we provided. Models will be dumped into `models` folder, and two csv files
will be generated, recording the information about results after cross validation and the real-prediction value of each room. Please keep these files since we'll use them to plot the figures.

We provide some demonstrations of the results by the XGBoost models.
![Accuracy Distribution Histogram](AccDis.png)
![RMSE Distribution Histogram](RMSEDis.png)

## 4. Result Visualization

After you've trained the models, run the `prediction_processing.py` to generate the visual graphs of the result. 

It will generate distribution plot for each room, interactive shapley value for each room's model, an overall RMSE histogram and an overall accuracy distribution histogram.

For detail about the Shapley value, please refer to [Shapley Additive Explanation](https://github.com/slundberg/shap)

The graphs will be dumped into three folders: `distribution_plot`, `shap_TH_ac_plot` and the current work directory.


## 5. Acknowledgement

This project is funded by Undergraduate Research Opportunities Program by HKUST UROP Office.

## 6. Citing this work

Please use the Bibtex below for citation of this work

```
Bibtex Citation to be updated
```
