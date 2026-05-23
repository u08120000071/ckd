# CHAPTER 3: METHODOLOGY

## 3.1 Research Design & Architectural Framework
The implementation of the Chronic Kidney Disease (CKD) diagnostic application is grounded in a hybrid architectural framework. It combines clinical-guideline heuristics with advanced machine learning classification to maximize diagnostic accuracy, safety, and clinical utility. Medical systems must maintain strict accountability. The system employs deterministic clinical criteria for staging, while using predictive classification algorithms to detect early-stage sub-clinical risk patterns. 

The process flow is divided into three distinct operational layers:
1. **Data Ingestion and Clinical Demographics Layer**: Manages patient registration, updates demographics, and collects biometric vitals and laboratory values.
2. **Deterministic & Predictive Diagnostic Pipeline**: 
    * Calculates the estimated Glomerular Filtration Rate (eGFR) and classifies the patient using KDIGO clinical staging (Deterministic).
    * Executes a Decision Tree Classifier trained on real patient dataset records to identify the predictive risk of CKD (Predictive).
    * Transmits clinical parameters, deterministic risk levels, and machine learning predictions to the Google Gemini large language model API to generate natural language clinical summaries and personalized recommendations.
3. **Application & Delivery Layer**: Delivers responsive web interfaces for clinical practitioners, supporting full patient lifecycle updates, clinical report generation, and interactive data visualization.

---

## 3.2 Dataset Selection and Distribution
The predictive model was trained on the Chronic Kidney Disease dataset from the University of California, Irvine (UCI) Machine Learning Repository. This dataset captures various clinical laboratory findings and biometric features for patients undergoing kidney disease evaluations.

For this implementation, the data was partitioned into a **training dataset** of 360 patient records and a **testing dataset** of 40 patient records (amounting to a 90/10 split). This structure was selected to maintain a distinct, isolated test cohort to rigorously evaluate model performance on unseen, clinical inputs.

### 3.2.1 Target Class and Feature Variables
The target variable is binary: `classification` represents the diagnostic status of the patient, mapped to `1` (CKD Detected) and `0` (Healthy / Not CKD). The target variable distribution in the training set contains:
*   **CKD Cases**: 225 records (62.5%)
*   **Healthy Cases**: 135 records (37.5%)

A carefully selected subset of 14 clinical features is captured by the diagnostic engine to support both deterministic staging and predictive modeling:

| Feature Name | Type | Unit | Description | Clinical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Age** | Numeric | Years | Patient age in years | Evaluated in the CKD-EPI staging equation. |
| **Blood Pressure** | Numeric | mmHg | Systolic / Diastolic indicator | Standard vital linked directly to renal vascular stress. |
| **Specific Gravity** | Numeric | - | Urine concentration | High Specific Gravity correlates with concentrated urine, a hallmark of renal stress. |
| **Albumin** | Categorical | Dipstick | Qualitative protein level (0 to 5) | Elevated albuminuria indicates glomerular filtration barrier leakage. |
| **Sugar Level** | Categorical | nominal | Urine sugar level (0 to 5) | Qualitative indicator for diabetes-associated osmotic stress. |
| **Red Blood Cells** | Categorical | Normal/Abnormal | Hematuria indicator | Sign of structural glomerular or tubular basement membrane damage. |
| **Blood Urea** | Numeric | mg/dL | Urea concentration in serum | Metabolic waste product whose retention signals clearance deficits. |
| **Serum Creatinine** | Numeric | mg/dL | Muscle breakdown byproduct | Primary endogenous biomarker for eGFR calculation. |
| **Sodium** | Numeric | mEq/L | Electrolyte balance | Disrupted electrolyte balance is common in progressive tubular dysfunction. |
| **Potassium** | Numeric | mEq/L | Electrolyte balance | Hyperkalemia risk increases severely as eGFR declines. |
| **Hemoglobin** | Numeric | g/dL | Red blood cell protein | Evaluated as an indicator of erythropoietin (EPO) synthesis deficits. |
| **White Blood Cell Count** | Numeric | cells/mcL | Immune cell count | Indicator of underlying acute/chronic interstitial nephritis or systemic inflammation. |
| **Hypertension** | Categorical | Yes/No | Clinical history | Co-morbid risk factor that damages afferent and efferent arterioles. |
| **Diabetes Mellitus** | Categorical | Yes/No | Clinical history | Leading global cause of diabetic nephropathy (hyperfiltration injury). |

---

## 3.3 Data Preprocessing and Imputation
Real-world clinical datasets are frequently characterized by missing records, which can severely compromise predictive performance. To resolve this, a robust, deterministic imputation and cleaning pipeline was constructed.

### 3.3.1 Handling Missing Values
Imputation parameters were learned strictly from the training dataset to prevent data leakage.
1.  **Numeric Features**: Missing numerical fields (e.g., Blood Urea, Hemoglobin, Sodium) are imputed using the **median value** of the respective cohort. Median imputation is selected over mean imputation to guard against distortion from clinical outliers.
2.  **Categorical Features**: Missing categorical parameters (e.g., Red Blood Cells, Hypertension) are imputed using the **mode** (most frequent class). If the cohort mode is unavailable, the clinical baseline standard ('no' for co-morbidities, 'normal' for RBC) is applied.

### 3.3.2 Categorical Variable Encoding
Categorical string inputs are converted into binary indicator variables to comply with input requirements of scikit-learn models:
*   `red_blood_cells` (rbc): Mapped to `1.0` if "normal", and `0.0` if "abnormal".
*   `hypertension` (htn): Mapped to `1.0` if "yes", and `0.0` if "no".
*   `diabetes_mellitus` (dm): Mapped to `1.0` if "yes", and `0.0` if "no".

---

## 3.4 The Diagnostic Processing Pipeline
The application executes a multi-stage, sequential clinical pipeline when evaluating a patient:

```
[Patient Clinical Vitals & Lab Inputs]
               │
               ▼
   [1. Preprocessing & Imputation]
               │
      ┌────────┴────────┐
      ▼                 ▼
[2. eGFR Equation]  [3. Decision Tree ML]
(CKD-EPI 2021)      (Inference & Confidence)
      │                 │
      ▼                 ▼
[4. KDIGO Staging]  [5. Feature Extraction]
(G1-G5 & A1-A3)         │
      │                 │
      └────────┬────────┘
               ▼
    [6. Gemini AI Engine]
    (Orchestration & Advice)
               │
               ▼
[PDF Report & Physician View]
```

### 3.4.1 Deterministic eGFR and KDIGO Staging
First, the engine extracts the patient's `serum_creatinine`, `age`, and `gender` to calculate the estimated Glomerular Filtration Rate using the modern, race-free **2021 CKD-EPI Creatinine Equation**:

$$	ext{eGFR} = 142 	imes \min\left(rac{S_{cr}}{\kappa}, 1ight)^{lpha} 	imes \max\left(rac{S_{cr}}{\kappa}, 1ight)^{-1.200} 	imes 0.9938^{	ext{Age}} 	imes 	ext{Sex\_Coeff}$$

Where:
*   $S_{cr}$ is the serum creatinine in mg/dL.
*   $\kappa$ is $0.7$ for females and $0.9$ for males.
*   $lpha$ is $-0.241$ for females and $-0.302$ for males.
*   $	ext{Sex\_Coeff}$ is $1.012$ for females and $1.0$ for males.

Using the computed eGFR and the Patient's Urine Albumin-to-Creatinine Ratio (ACR), the system classifies the patient's renal status according to the clinical standard **KDIGO Guidelines**:

1.  **GFR Staging (G1 to G5)**:
    *   **G1**: $\ge 90$ mL/min/1.73 m² (Normal or high)
    *   **G2**: $60 - 89$ mL/min/1.73 m² (Mildly decreased)
    *   **G3a**: $45 - 59$ mL/min/1.73 m² (Mildly to moderately decreased)
    *   **G3b**: $30 - 44$ mL/min/1.73 m² (Moderately to severely decreased)
    *   **G4**: $15 - 29$ mL/min/1.73 m² (Severely decreased)
    *   **G5**: $< 15$ mL/min/1.73 m² (Kidney failure)

2.  **Albuminuria Staging (A1 to A3)**:
    *   **A1**: $< 30$ mg/g (Normal to mildly increased)
    *   **A2**: $30 - 300$ mg/g (Moderately increased)
    *   **A3**: $> 300$ mg/g (Severely increased)
    *   *Note*: If the raw laboratory ACR is not supplied, the system dynamically estimates it based on qualitative dipstick urine albumin: `0` $ightarrow 15.0$ mg/g, `1` $ightarrow 150.0$ mg/g, $\ge 2$ $ightarrow 450.0$ mg/g.

3.  **Risk Progression Matrix**: The system maps the GFR stage and Albuminuria category onto the clinical KDIGO heatmap, producing a final prognostic risk category: **Low, Moderate, High, or Very High**, along with institutional recommendations.

### 3.4.2 Machine Learning Classification
Simultaneously, the patient's 14 features are packaged as an input vector and evaluated by a **Decision Tree Classifier**. The model determines a diagnostic binary prediction (CKD Detected or No CKD Detected) and extracts the leaf-node class distribution to calculate a prediction **confidence score**. 

### 3.4.3 Large Language Model Clinical Summarization
To provide personalized care recommendations, the results are forwarded to the **Google Gemini Pro API**. The orchestration script constructs a clinical prompt containing the patient’s vital statistics, clinical staging, ML classifier prediction, and confidence score. The model returns structured clinical insights, risk assessments, suggested precautions, lifestyle modifications, and follow-up protocols.

---

## 3.5 System Architecture & Implementation
The application is built on a clean, scalable Model-View-Controller (MVC) pattern using Flask (Python) and SQLite/PostgreSQL databases.

```
       [ Client Web Browser ]
         │               ▲
  (HTTP Requests)  (Rendered HTML/CSS)
         ▼               │
    [ app/routes/dashboard.py ]  ◄── (Flask Blueprints)
         │               ▲
         ▼               │
   [ app/forms.py ] [ app/templates/ ]
         │
         ├───► [ app/models.py ] ───► [ PostgreSQL / SQLite DB ]
         │
         ├───► [ app/diagnosis.py ] (eGFR & KDIGO Calculations)
         │
         ├───► [ app/ml_model.py ] ──► [ decision_tree.joblib ]
         │
         └───► [ app/gemini_service.py ] ──► [ Gemini Pro API ]
```

*   **Models (`app/models.py`)**: Utilizes SQLAlchemy to define relational schemas for `User` (clinicians), `Patient` (demographics), and `Diagnosis` (historical laboratory metrics).
*   **Views (`app/templates/`)**: Uses custom Jinja2 HTML templates styled with premium vanilla CSS variables. Pages are fully responsive and utilize clean micro-animations.
*   **Controllers (`app/routes/`)**: Implements Blueprint controllers to manage application authentication, patient registers, and diagnosis forms.
*   **Interactive Editing**: Includes dedicated edit controllers that allow clinical practitioners to update patient demographics or retroactively modify historical clinical inputs, triggering automated diagnostic re-runs to keep patient records consistent.

---
page-break

# CHAPTER 4: RESULTS AND DISCUSSION

## 4.1 Experimental Environment & Setup
To establish reproducibility, the system development and model evaluations were executed in the following localized standard environment:
*   **Operating System**: Linux Ubuntu 24.04 LTS
*   **Core Execution Framework**: Python 3.12.3
*   **Primary Scientific Computing Libraries**:
    *   `scikit-learn` (v1.4.2) for Decision Tree modeling, validation, and feature importances.
    *   `pandas` (v2.2.1) and `numpy` (v1.26.4) for dataset manipulation, filtering, and numerical vector operations.
    *   `joblib` (v1.4.0) for predictive model serialization and caching.
    *   `Flask` (v3.0.2) and `Flask-SQLAlchemy` (v3.1.1) for web deployment and data persistence.
    *   `weasyprint` (v61.2) for compiling HTML reports into print-ready PDF assets.

---

## 4.2 Model Training Performance
A Decision Tree Classifier was selected due to its high clinical interpretability. Unlike "black box" deep neural networks, a Decision Tree's path corresponds to a sequence of logical medical rule splits that can be manually traced and validated by a physician.

The classifier was trained on the cleaned 360-sample training dataset. Regularization was implemented using a `min_samples_leaf=2` parameter to ensure leaf nodes capture generalizable clinical sub-cohorts rather than isolated patients, reducing overfitting.

The training execution returned the following architectural metrics:
*   **Optimal Tree Depth**: 6 splits
*   **Total Terminating Leaf Nodes**: 11
*   **Training Set Accuracy**: **99.72%**
*   **5-Fold Stratified Cross-Validation Accuracy**: **95.56% (±3.66%)**

The extremely high cross-validation performance confirms that the model generalizes robustly across different sub-cohort partitions of the primary dataset, maintaining strong performance outside of its initial training set.

---

## 4.3 Feature Importance Analysis
Analyzing feature importances in clinical models is highly valuable. It reveals which physiological indicators have the greatest predictive power for diagnosing Chronic Kidney Disease. The Gini feature importances calculated by the Decision Tree model are outlined below:

| Feature Variable | Gini Importance | Cumulative Significance | Description |
| :--- | :--- | :--- | :--- |
| **Hemoglobin (`hemo`)** | **0.7177 (71.77%)** | 71.77% | Red blood cell oxygen carrier protein |
| **Specific Gravity (`sg`)** | **0.1950 (19.50%)** | 91.27% | Urinalysis concentration ratio |
| **Albumin (`al`)** | **0.0235 (2.35%)** | 93.62% | Dipstick urine protein indicator |
| **Blood Urea (`bu`)** | **0.0227 (2.27%)** | 95.89% | Nitrogenous waste byproduct |
| **Age (`age`)** | **0.0218 (2.18%)** | 98.07% | Patient chronological age |
| **Serum Creatinine (`sc`)** | **0.0194 (1.94%)** | 100.00% | Primary muscular waste biomarker |

### 4.3.1 Clinical Discussion of Primary Predictive Features
1.  **Hemoglobin (71.77%)**: The dominant predictive value of hemoglobin is highly aligned with clinical pathophysiology. As chronic kidney disease progresses, structural damage to the renal interstitium degrades the specialized peritubular capillary cells responsible for producing **Erythropoietin (EPO)**. A drop in EPO synthesis reduces red blood cell production, leading to **renal anemia**. Since this decline is directly tied to the functional mass of the kidneys, hemoglobin is an exceptionally strong, early physiological marker of chronic kidney damage.
2.  **Specific Gravity (19.50%)**: This indicator reflects the concentration-dilution capacity of the renal tubules. Tubular atrophy and interstitial fibrosis impair the kidneys' ability to concentrate urine. This leads to a persistently fixed specific gravity (isosthenuria, typically around 1.010), representing a clear indicator of structural tubular injury.
3.  **Albumin (2.35%)**: Urinalysis findings showing persistent leakage of albumin into urine are direct indicators of structural damage to the glomerular filtration barrier. It highlights basement membrane damage and podocyte depletion.

---

## 4.4 Model Validation and Evaluation on Test Set
To validate the model's accuracy on unseen clinical cases, a separate **test dataset** of 40 patient records was evaluated. This cohort contained 25 diagnosed CKD patients and 15 healthy controls.

The model achieved highly successful results, returning the following performance metrics:
*   **Test Classification Accuracy**: **97.50%**
*   **Precision (Positive Predictive Value)**: **100.00%**
*   **Recall / Sensitivity**: **96.00%**
*   **Specificity (True Negative Rate)**: **100.00%**
*   **F1-Score**: **97.96%**

### 4.4.1 Confusion Matrix Analysis
The confusion matrix illustrates the distribution of true versus predicted classifications:

| Predicted Class | Actual CKD Positive (25) | Actual Healthy (15) |
| :--- | :--- | :--- |
| **Predicted CKD** | **24 (True Positive - TP)** | **0 (False Positive - FP)** |
| **Predicted Healthy** | **1 (False Negative - FN)** | **15 (True Negative - TN)** |

### 4.4.2 Clinical Interpretation of Metrics
*   **Perfect Specificity (100.00%) & Precision (100.00%)**: The model generated **zero False Positives**. In a clinical setting, this is highly valuable as it prevents healthy individuals from experiencing unnecessary emotional stress, invasive diagnostic follow-ups, and unwarranted treatment costs.
*   **High Sensitivity (96.00%)**: The model successfully identified 24 out of 25 true CKD cases, producing only a single False Negative. In screening software, maintaining high sensitivity is critical to ensure patients with progressive, silent diseases are not missed during routine clinical checkups.

---

## 4.5 System Demonstration & Case Studies
The clinical value of the system is demonstrated through its execution of diverse, real-world patient case studies, combining deterministic staging, predictive model scoring, and Gemini-guided clinical advice.

### 4.5.1 Diagnostic Case Study 1: Early-Stage Diabetic Nephropathy
*   **Patient Profile**: Female, 54 years old, history of diabetes mellitus.
*   **Laboratory Values**: Serum Creatinine: 1.1 mg/dL, Urine ACR: 120.0 mg/g (Albumin Dipstick: 1+).
*   **Deterministic Evaluation**:
    *   Computed eGFR: 69.17 mL/min/1.73 m² (GFR Stage: **G2**, Mildly decreased).
    *   Albuminuria Category: **A2** (Moderately increased).
    *   KDIGO Staging: **G2/A2** $ightarrow$ **Moderate Risk**.
*   **Machine Learning Classifier Evaluation**:
    *   Prediction: **CKD Detected**
    *   Inference Confidence: **100.0%**
*   **AI Advisory Output**: The Gemini clinical assistant highlighted that while the eGFR indicates only mild functional decline (G2), the presence of moderate albuminuria (A2) in a diabetic patient strongly suggests early-stage diabetic nephropathy. The system recommended initiating an ACE inhibitor or Angiotensin Receptor Blocker (ARB) for renal protection, maintaining tight blood glucose control (HbA1c < 7.0%), and scheduling follow-up monitoring in 6 months.

### 4.5.2 Diagnostic Case Study 2: Advanced-Stage Renal Failure
*   **Patient Profile**: Male, 68 years old, history of severe hypertension.
*   **Laboratory Values**: Serum Creatinine: 4.8 mg/dL, Urine ACR: 450.0 mg/g (Albumin Dipstick: 3+), Hemoglobin: 8.4 g/dL.
*   **Deterministic Evaluation**:
    *   Computed eGFR: 12.35 mL/min/1.73 m² (GFR Stage: **G5**, Kidney Failure).
    *   Albuminuria Category: **A3** (Severely increased).
    *   KDIGO Staging: **G5/A3** $ightarrow$ **Very High Risk**.
*   **Machine Learning Classifier Evaluation**:
    *   Prediction: **CKD Detected**
    *   Inference Confidence: **100.0%**
*   **AI Advisory Output**: The system issued an urgent alert highlighting advanced kidney failure (Stage G5) combined with severe anemia (Hemoglobin 8.4 g/dL), which is a common systemic complication of chronic kidney disease. The clinical assistant recommended an immediate referral to nephrology, preparing a vascular access plan for hemodialysis, and evaluating the patient for erythropoietin-stimulating agents (ESAs) to treat renal anemia.

---

## 4.6 Clinical Discussion & Future Outlook
The results demonstrate that combining deterministic clinical staging with predictive machine learning provides a highly effective diagnostic screening tool. 

The high predictive accuracy (97.50% test accuracy) achieved by a simple, interpretable Decision Tree highlights the value of focusing on key clinical biomarkers like Hemoglobin, Specific Gravity, and Albumin. Rather than using overly complex "black-box" models, this approach delivers a clear, rule-based clinical pipeline that can be easily explained and verified by healthcare professionals.

### 4.6.1 System Limitations
1.  **Imputation Assumptions**: While median and mode imputation are effective for general population screening, they can occasionally mask unique physiological variations in highly atypical patients.
2.  **eGFR Formula Constraints**: The 2021 CKD-EPI equation is highly optimized for steady-state chronic conditions but is less reliable in acute clinical settings (such as acute kidney injury, AKI) where creatinine levels fluctuate rapidly.

### 4.6.2 Future Outlook
Future iterations of this platform will focus on integrating longitudinal patient data to model the **rate of eGFR decline** over time. This will allow the system to predict the trajectory of kidney disease progression, giving clinicians the insights needed to implement preventative therapies even earlier.
