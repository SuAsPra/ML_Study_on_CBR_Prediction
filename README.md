# CBR Prediction of Sand/Soil Using Machine Learning

This project is a complete machine learning study for estimating the California Bearing Ratio (CBR) of soil using commonly available geotechnical index properties. The final trained model is deployed through a Python backend and a React frontend application, where the user enters known laboratory values and receives an estimated CBR value, an expected prediction range, and SHAP-based feature influence.

The final deployed model is focused on cohesive soils because the available dataset showed that cohesive soil records were large enough and consistent enough to produce a reliable model. The app can still accept UCSCS text for warning logic, but the scientific model scope is cohesive soil behavior.

## Project Objective

CBR is an important geotechnical parameter used in pavement design and subgrade strength evaluation. Traditional CBR testing is time-consuming because it requires sample preparation, soaking/conditioning in some cases, penetration testing, and careful laboratory control.

The objective of this study was to build an ML-based estimation model that can predict CBR from easier-to-obtain soil properties:

- LL (%)
- PL (%)
- OMC (%)
- MDD (g/cm3)
- Gravel (%)
- Sand (%)
- Fines

In the final frontend application, all seven of these values are compulsory because they are part of the deployed final model input space.

## Quick Run

Run everything from the project root using the existing virtual environment.

Start the Python backend:

```powershell
.\venv\Scripts\python.exe .\cbr-react-app\backend\server.py
```

Backend URL:

```text
http://localhost:8000
```

Open a second terminal and start the React frontend:

```powershell
cd .\cbr-react-app
npm.cmd install
npm.cmd run dev
```

Frontend URL:

```text
http://localhost:5173
```

Use the frontend form to enter the compulsory values: LL, PL, OMC, MDD, Gravel, Sand, and Fines.

## Repository Structure

```text
.
├── step1_load_and_inspect.py
├── step2_fix_structure.py
├── step3_build_base_dataset.py
├── step4_eda.py
├── step5_ml_models.py
├── step5_ml_models_new.py
├── step6_improve_model.py
├── step6_cv_model.py
├── step6_cv_fixed.py
├── step7_hybrid_model.py
├── step8_soil_split_model.py
├── step9_feature_engineering.py
├── step10_log_transform.py
├── step11_residual_filtering.py
├── step12_save_final_model.py
├── step13_residual_filtering_with_pl.py
├── step14_save_final_model_with_pl.py
├── final_cbr_model_with_pl.pkl
├── imputer_with_pl.pkl
├── features_with_pl.pkl
└── cbr-react-app/
    ├── backend/server.py
    └── src/App.jsx
```

## Environment Setup

All Python runs should be maintained inside the virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl xgboost shap
```

The project was run with:

```text
pip 25.3
Python 3.14 virtual environment
```

## Theory

### California Bearing Ratio

CBR measures the resistance of soil to penetration compared with a standard crushed rock material. In pavement and subgrade design, higher CBR generally means stronger bearing capacity and better support for pavement layers. Low CBR values indicate weak subgrade material that may need stabilization, replacement, compaction control, or increased pavement thickness.

### Soil Parameters Used

Liquid Limit (LL) is the water content at which soil changes from plastic to liquid behavior. A high LL often indicates high clay activity or high water sensitivity.

Plastic Limit (PL) is the water content at which soil changes from semi-solid to plastic behavior. PL helps describe the consistency behavior of fine-grained soil.

Plasticity Index (PI) is usually calculated as `PI = LL - PL`. PI was studied during model development, but the final deployed model uses PL directly along with LL instead of relying only on PI.

Optimum Moisture Content (OMC) is the water content at which soil reaches maximum dry density under a given compaction effort. OMC is strongly connected with compaction behavior and strength.

Maximum Dry Density (MDD) represents the maximum compacted dry unit density of soil. Higher density generally improves particle interlock and bearing strength, although soil type and moisture condition also matter.

Gravel, Sand, and Fines describe the particle-size distribution. These features are important because coarse particles, sand fraction, and fine particles affect drainage, compaction, plasticity, and strength.

### Why Machine Learning Is Suitable

CBR depends on nonlinear interactions between moisture, density, plasticity, and gradation. A simple linear equation cannot fully capture this behavior. For example, PI had almost zero linear correlation with CBR in early analysis, but tree-based feature importance showed that plasticity-related variables can still influence prediction through nonlinear interactions.

This is why tree-based ensemble models such as Random Forest and XGBoost were tested. They can learn threshold effects, interactions, and nonlinear relationships without manually defining every physical equation.

## Dataset Cleaning and Preparation

### Step 1: Load and Inspect

The raw dataset had 2089 rows and 15 columns. The first row contained actual subheadings, so the cleaning process removed the extra heading row and retained proper column names.

Output:

```text
Shape: (2089, 15)
step1_cleaned.xlsx created
```

### Step 2: Fix Structure

The cleaned dataset was standardized into meaningful column names:

```text
LL_, PL_, PI_, OMC_, MDDg/cm3, AASHTO, UCSCS,
SPECIFIC_GRAVITY_g/cm3, Gravel_, Sand_, Fines, CBR_, doi
```

Feature availability was checked:

| Availability | Features |
| --- | --- |
| Strong | LL, PI, OMC, MDD, AASHTO, UCSCS, CBR |
| Medium | PL, Fines |
| Weak | Gravel, Sand, Specific Gravity, DOI |

At this stage, weak features were initially dropped to avoid excessive missing-data problems.

### Step 3: Base Dataset

A base dataset was created using the stronger numerical features:

```text
Shape: (1370, 5)
step3_base_dataset.xlsx created
```

### Step 4: Exploratory Data Analysis

After numeric cleaning and removal of unrealistic values:

```text
Original shape: (1370, 5)
After numeric cleaning: (1369, 5)
After removing unrealistic values: (1036, 5)
```

Main statistics:

| Feature | Mean | Standard Deviation |
| --- | ---: | ---: |
| LL | 31.28 | 21.16 |
| PI | 13.14 | 10.67 |
| OMC | 13.68 | 8.74 |
| MDD | 1.52 | 0.50 |
| CBR | 9.72 | 13.77 |

Correlation with CBR:

| Feature | Correlation with CBR |
| --- | ---: |
| OMC | 0.356 |
| MDD | 0.298 |
| LL | 0.291 |
| PI | 0.018 |

Interpretation:

- OMC was the strongest linear predictor.
- MDD and LL moderately influenced CBR.
- PI had almost no linear correlation.
- The weak PI correlation did not necessarily mean PI or plasticity was useless, because correlation only measures linear relationship.

## Model Development

### Step 5: Baseline ML Models

Three regression models were tested:

| Model | R2 | RMSE |
| --- | ---: | ---: |
| Linear Regression | 0.27 | 10.00 |
| Random Forest | 0.53 | 8.06 |
| XGBoost | 0.60 | 7.38 |

Result:

XGBoost performed best. This confirmed that CBR prediction is nonlinear and tree-based ensemble models are more suitable than simple linear regression.

### Step 5B: With PI vs Without PI

XGBoost was tested with and without PI:

| Model Case | XGBoost R2 |
| --- | ---: |
| With PI | 0.60 |
| Without PI | 0.64 |

Although PI showed importance in tree-based models, removing it improved performance. This suggested that PI may introduce redundancy or noise because `PI = LL - PL`, so much of its information overlaps with LL and PL.

### Step 6: Advanced XGBoost

An improved XGBoost model was attempted:

```text
RMSE: 7.75
R2: 0.56
```

This reduced performance. The conclusion was important:

More complex model tuning does not always improve performance. With noisy, heterogeneous, multi-source soil data, better data quality and better features matter more than simply increasing model complexity.

### Step 6: Cross Validation

Initial K-Fold cross validation produced unstable results:

```text
R2 scores: [-2.97, -0.72, 0.17, -0.02, 0.66]
Mean R2: -0.57
```

This happened because random folds had uneven CBR distributions and the dataset came from different papers, soil types, and testing conditions.

After fixing the validation approach with stratified splitting:

```text
R2 scores: [0.623, 0.772, 0.704, 0.770, 0.825]
Mean R2: 0.739
Std Dev: 0.070
```

This showed the model was much more reliable when the folds were balanced.

## Missing Data Strategy

### Step 7: Hybrid Modeling With Imputation

The study then returned to the wider dataset and tested whether missing gradation features could still be useful.

| Case | Mean R2 |
| --- | ---: |
| Without Imputation | 0.677 |
| With Median Imputation | 0.741 |

Median imputation improved dataset utilization and performance. The missing gradation features were useful, and the missingness itself appeared informative.

GAIN or deep-learning-based imputation was considered but rejected because it could generate artificial realistic-looking data and inflate R2 in a scientifically unsafe way.

## Soil-Type-Specific Modeling

### Step 8: UCSCS-Based Split

The dataset was split by soil type:

```text
Cohesive: 1866
Granular: 177
Mixed: 45
```

Model performance:

| Soil Type | Mean R2 | Std Dev |
| --- | ---: | ---: |
| Cohesive | 0.881 | 0.021 |
| Granular | 0.608 | 0.344 |

The cohesive model was strong and stable. The granular model was unstable because the granular dataset was too small, too diverse, and poorly represented.

Final decision:

The final model focuses on cohesive soils. This is a scientifically safer decision than forcing one model to represent all soil types when the data is imbalanced.

## Feature Engineering

### Step 9: Physics-Based Features

Domain-informed features were created:

| Feature | Meaning |
| --- | --- |
| OMC/MDD | moisture-density interaction |
| LL/MDD | plasticity-density interaction |
| LL x OMC | combined plasticity and moisture behavior |

Result:

```text
Mean R2: 0.886
Std Dev: 0.024
```

The improvement from 0.881 to 0.886 was small, but the features were physically meaningful and were retained.

### Step 10: Log Transformation

Log transformation was tested because it can sometimes reveal hidden patterns in nonlinear datasets. It failed for this case:

```text
R2 dropped to about 0.875
```

So log transformation was rejected.

## Final Model Boost

### Step 11: Residual Filtering

The model was already performing well, but some noisy samples were causing high errors. A residual-filtering approach was used:

1. Train an initial model.
2. Predict on the training data.
3. Calculate residual error.
4. Remove the worst 10% high-error samples.
5. Retrain the model.

Output:

```text
Original size: 1866
Filtered size: 1679
R2 scores: [0.943, 0.841, 0.899, 0.927, 0.939]
Mean R2: 0.910
Std Dev: 0.038
```

Final interpretation:

By combining soil-type-specific modeling, feature engineering, median imputation, and residual-based data refinement, the final cohesive-soil model achieved about 91% cross-validated R2, with the best fold reaching about 94%.

Residual filtering must be justified carefully. It is not used to hide poor results; it is used because multi-source experimental soil data can contain inconsistent testing conditions, reporting errors, and outlier samples that reduce model reliability.

## Final Model Artifacts

The final deployed model with PL was saved using:

```powershell
.\venv\Scripts\python.exe .\step14_save_final_model_with_pl.py
```

Created files:

```text
final_cbr_model_with_pl.pkl
imputer_with_pl.pkl
features_with_pl.pkl
```

Final deployed features:

```text
LL
PL
OMC
MDD
Fines
Sand
Gravel
OMC_MDD
LL_MDD
LL_OMC
```

The user enters LL, PL, OMC, MDD, Fines, Sand, and Gravel. The backend computes the engineered features automatically.

## Final Model Result

Final selected model:

```text
XGBoost Regressor
Soil group: Cohesive soils
Mean R2: approximately 0.91
Best fold R2: approximately 0.94
Prediction output limit: CBR 0 to 100
```

The final result can be stated as:

Machine-learning-based CBR prediction is strongly nonlinear. A soil-specific XGBoost model trained on cohesive soils, supported by median imputation, physics-based feature engineering, and residual-based refinement, achieved a mean R2 of approximately 0.91. This indicates that accurate CBR estimation is possible when the model is restricted to a consistent soil group and uses domain-informed preprocessing.

## Frontend and Backend App

The final model is used in a React + Python application.

Backend:

```text
cbr-react-app/backend/server.py
```

Frontend:

```text
cbr-react-app/src/App.jsx
cbr-react-app/src/App.css
cbr-react-app/src/main.jsx
```

The app provides:

- CBR prediction
- CBR range estimate
- CBR clamped between 0 and 100
- SHAP explanation of the top factors affecting prediction
- Warning when UCSCS input does not appear to be cohesive
- Compulsory input validation for LL, PL, OMC, MDD, Gravel, Sand, and Fines

## Running the Application

Start the backend from the project root:

```powershell
.\venv\Scripts\python.exe .\cbr-react-app\backend\server.py
```

The backend runs at:

```text
http://localhost:8000
```

Start the frontend in a second terminal:

```powershell
cd .\cbr-react-app
npm.cmd install
npm.cmd run dev
```

The frontend runs at:

```text
http://localhost:5173
```

## API Endpoints

Health check:

```text
GET /health
```

Feature list:

```text
GET /features
```

Prediction:

```text
POST /predict
```

Example payload:

```json
{
  "ll": 42,
  "pl": 21,
  "omc": 14.5,
  "mdd": 1.78,
  "gravel": 5,
  "sand": 42,
  "fines": 53,
  "ucscs": "CL"
}
```

## Important Scientific Notes

- The final model is intended for cohesive soils.
- Granular soil prediction was not selected because granular data was limited and unstable.
- PL is compulsory in the deployed final model.
- PI is accepted by the UI as optional but is not used by the final deployed model.
- AASHTO, UCSCS, and Specific Gravity are accepted for context, but they are not direct numerical inputs to the final XGBoost model.
- UCSCS is used for cohesive-soil warning logic.
- The model is an estimation tool, not a replacement for laboratory CBR testing.
- Predictions should be used for preliminary study, comparison, and decision support.

## Main Research Conclusions

1. CBR prediction is nonlinear.
2. XGBoost outperformed Linear Regression and Random Forest.
3. Correlation alone was not enough to judge feature usefulness.
4. PI showed weak linear correlation and could introduce redundancy.
5. Median imputation improved use of partially missing gradation features.
6. Cohesive-soil-specific modeling was much more reliable than mixed-soil modeling.
7. Physics-based features gave a small but meaningful domain-supported improvement.
8. Log transformation did not improve the model.
9. Residual filtering improved final reliability by reducing the effect of noisy multi-source samples.
10. The final cohesive-soil model achieved approximately 91% R2.
#   M L _ S t u d y _ o n _ C B R _ P r e d i c t i o n  
 