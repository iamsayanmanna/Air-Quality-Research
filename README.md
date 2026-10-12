# 🌍 When Air Quality Models Fail
### Robustness of Machine Learning Predictions During Extreme Pollution Events in Indian Cities

<p align="center">
  <strong>Can machine learning models remain reliable when air pollution reaches extreme levels?</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Research-Air%20Quality-0D9488?style=for-the-badge" alt="Air Quality Research" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Model%20Evaluation-2563EB?style=for-the-badge" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/Deep%20Learning-Time%20Series-7C3AED?style=for-the-badge" alt="Deep Learning" />
  <img src="https://img.shields.io/badge/Status-Research%20Project-F59E0B?style=for-the-badge" alt="Research Project" />
</p>

---

## 📌 Overview

Air pollution prediction is an important application of data science, but a model that performs well under ordinary conditions may not perform equally well during extreme pollution episodes.

This research investigates the **robustness and reliability of machine learning and deep learning models during extreme air pollution events in Indian cities**.

Using historical air-quality data, the project explores pollution patterns, develops predictive models, evaluates their performance on chronological test data, and examines how prediction errors behave during extreme pollution conditions.

Rather than evaluating models only through overall performance, this study places special emphasis on their ability to handle high-pollution observations.

> **Core research question:** Do air-quality prediction models remain reliable during extreme pollution episodes, or does their performance deteriorate when accurate predictions matter most?

## 🎯 Research Objectives

- Perform exploratory data analysis (EDA) to understand air-quality patterns, distributions, and extreme observations.
- Develop machine learning models for air-quality prediction.
- Implement and compare deep learning architectures for time-series prediction.
- Evaluate model performance using appropriate regression metrics.
- Investigate model behaviour during extreme pollution episodes.
- Compare overall test performance with performance on extreme-pollution observations.
- Identify the strengths and limitations of different model architectures.

## 🔬 Research Design

The workflow follows a structured, time-aware experimental approach:

1. **Exploratory Data Analysis** — investigate data quality, distributions, temporal patterns, and relationships between variables.
2. **Data Preparation** — define the prediction target, select features, handle missing target values, encode categorical variables, and scale features.
3. **Chronological Evaluation** — preserve temporal ordering when splitting training and testing data.
4. **Model Development** — train machine learning and deep learning models.
5. **Overall Evaluation** — assess predictions on the chronological test set.
6. **Extreme-Event Evaluation** — evaluate predictions for observations meeting the predefined extreme-pollution threshold.
7. **Comparative Analysis** — compare overall metrics, extreme-event errors, and the change in error metrics across evaluation subsets.

## 📂 Project Structure

```text
Air-Quality-Research/
│
├── notebooks/
│   ├── 01_EDA_Air_Quality.ipynb
│   ├── 02_Machine_Learning_Air_Quality.ipynb
│   └── 03_Deep_Learning_Air_Quality.ipynb
│
├── dataset/
│   └── INDIA_AQI_COMPLETE_20251126.csv
│       └── Kept locally; not tracked by Git
│
├── graphs/          # Local research visualizations
├── results/         # Local research outputs
├── .gitignore
└── README.md
```

*The dataset, graphs, and results directories are part of the local working project. Only files explicitly committed to Git are available in this GitHub repository.*

## 📓 Research Notebooks

| Notebook | Description |
|---|---|
| [01 — Exploratory Data Analysis](notebooks/01_EDA_Air_Quality.ipynb) | Data exploration, quality checks, statistical summaries, and visual analysis |
| [02 — Machine Learning](notebooks/02_Machine_Learning_Air_Quality.ipynb) | Machine learning experiments and model evaluation |
| [03 — Deep Learning](notebooks/03_Deep_Learning_Air_Quality.ipynb) | MLP, RNN, GRU, and Transformer experiments, evaluation, robustness analysis, and visualizations |

Open the notebooks above to explore the code, analysis, experiments, and recorded outputs.

## 📊 Dataset

**Source:** [Air Quality Dataset — Indian Cities (2022–2025), Kaggle](https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025/data)

The local dataset used in the experiments is named `INDIA_AQI_COMPLETE_20251126.csv`.

The working dataset contains approximately **842,160 rows across 71 columns**, covering 29 cities and the period 2022–2025.

The dataset is not included in this GitHub repository because the local CSV is approximately 282.6 MB. Please obtain the data from the original Kaggle source and comply with its applicable access and licensing terms.

After downloading the dataset, place it at:

```text
dataset/INDIA_AQI_COMPLETE_20251126.csv
```

## ⚙️ Experimental Setup

| Component | Configuration |
|---|---|
| Prediction target | `US_AQI` |
| Extreme-pollution threshold | `US_AQI >= 201` |
| Geographic coverage | 29 Indian cities |
| Time period | 2022–2025 |
| Data split | Chronological 80% training / 20% testing |
| Time-series lookback window | 24 time steps |
| Input features for deep learning | 14 |
| Parameters per deep learning model | 76,601 |
| Maximum training epochs | 20 |
| Batch size | 512 |
| Validation split | 10% of the training sequences |
| Early stopping patience | 5 epochs |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Mean Squared Error (MSE) |
| Random seed | 42 |

The deep learning experiments compare four architectures with the same parameter count:

- **MLP** — Multilayer Perceptron
- **RNN** — Simple Recurrent Neural Network
- **GRU** — Gated Recurrent Unit
- **Transformer** — Attention-based sequence model

Equal parameter counts provide a more controlled comparison of model size, but do not make the architectures identical in computational cost, inductive bias, or effective learning capacity.

## 📏 Evaluation Metrics

The project evaluates regression performance using:

- **MAE — Mean Absolute Error:** average absolute prediction error.
- **RMSE — Root Mean Squared Error:** penalizes large prediction errors more heavily.
- **R² — Coefficient of Determination:** measures performance relative to a mean-based baseline; negative values indicate performance worse than that baseline on the evaluated sample.

Metrics are reported separately for the full chronological test set and the extreme-pollution subset.

## 🧠 Deep Learning Results

The following values are taken from the recorded model-comparison output in the current deep learning notebook.

| Model | Parameters | Overall MAE | Overall RMSE | Overall R² | Extreme MAE | Extreme RMSE | Extreme R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 76,601 | 11.6993 | 50.1122 | 0.8014 | 208.2278 | 307.9689 | 0.5115 |
| RNN | 76,601 | 21.6034 | 75.3754 | 0.5507 | 297.3423 | 436.7031 | 0.0178 |
| GRU | 76,601 | 15.4666 | 54.1908 | 0.7678 | 194.6090 | 319.3998 | 0.4746 |
| Transformer | 76,601 | 12.2016 | 49.9385 | 0.8028 | 196.4280 | 303.3882 | 0.5260 |

### Test-set composition

- **Total test samples:** 168,432
- **Extreme-pollution samples (`US_AQI >= 201`):** 4,324
- **Extreme-pollution share of test set:** 2.57%

### Key Observations

- **Lowest overall MAE:** MLP, at **11.6993**.
- **Best overall RMSE and overall R²:** Transformer, with RMSE **49.9385** and R² **0.8028**.
- **Lowest extreme-pollution MAE:** GRU, at **194.6090**.
- **Best extreme-pollution RMSE and R²:** Transformer, with RMSE **303.3882** and R² **0.5260**.
- The RNN had the highest extreme-pollution MAE (**297.3423**) and a near-zero extreme-subset R² (**0.0178**).
- The best-performing architecture depends on the metric and evaluation subset; overall performance alone does not fully describe performance during extreme pollution.

### Robustness Interpretation

In the notebook's comparison of overall MAE with extreme-subset MAE, the GRU has the lowest extreme-to-overall MAE ratio, approximately **12.58×**. This is a descriptive comparison between different subsets of the test data, not a standalone proof of model robustness: the subsets have different target distributions and sample sizes. The extreme subset contains only 2.57% of the test observations, so these results should be interpreted with that imbalance in mind.

These are experiment-specific results. They do not establish that the same model ranking will hold for other datasets, cities, temporal splits, or configurations.

## 🌫️ Why Extreme Pollution Matters

Extreme pollution observations may be relatively uncommon, yet prediction errors during these episodes can be especially consequential.

This research therefore considers more than average model performance. It asks whether models can predict unusually high pollution values and how their errors behave under the conditions that are the primary focus of the study.

The objective is to encourage careful evaluation of predictive systems intended for environmental monitoring and pollution-related decision support.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow / Keras
- **Development Environment:** Jupyter Notebook
- **Version Control:** Git and GitHub
- **Dataset Source:** Kaggle

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/iamsayanmanna/Air-Quality-Research.git
cd Air-Quality-Research
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

### 3. Install the required libraries

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter tensorflow
```

*Package versions may affect reproducibility. Pin the versions used for the final experiments in a requirements file when preparing a reproducible release.*

### 4. Download the dataset

Download the dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025/data) and place the CSV at the expected local path described in the Dataset section.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory and execute them in the documented order, beginning with EDA.

**Note:** The notebooks may rely on local dataset paths and saved outputs. Verify the required paths and dependencies before running them in a fresh environment.

## ⚠️ Limitations and Responsible Interpretation

- Results depend on the dataset, preprocessing choices, feature selection, temporal split, and model configuration.
- A chronological holdout evaluates later observations from the available data but does not automatically establish generalization to entirely unseen cities or future pollution regimes.
- The extreme-pollution threshold is an operational research definition for this experiment and should be interpreted according to the target index and its applicable scale.
- Extreme-pollution observations make up a small fraction of the test set, so subset metrics may be sensitive to the limited sample.
- Equal parameter counts do not guarantee equal training cost or equivalent model capacity.
- Comparing overall and extreme-subset metrics requires care because their target distributions and sample sizes differ.
- Predictive performance alone does not establish causal relationships between pollutants, weather, and air quality.
- These experiments are research results and should not be treated as a substitute for official air-quality monitoring or public-health guidance.

## 🔭 Future Work

- Compare extreme-pollution performance with a clearly defined non-extreme subset using consistent metrics.
- Investigate performance across cities, seasons, and individual pollution episodes.
- Evaluate calibration, prediction intervals, and uncertainty during extreme events.
- Test alternative temporal validation strategies and out-of-city generalization.
- Compare additional baselines and conduct systematic hyperparameter tuning.
- Improve reproducibility through pinned dependencies, documented configurations, and versioned experiment outputs.

## 👨‍💻 Author

**Sayan Manna**

BCA Student | Data Analytics | Machine Learning | Deep Learning | Web Development

- **GitHub:** [@iamsayanmanna](https://github.com/iamsayanmanna)
- **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/iamsayanmanna/)

## 📚 Dataset Attribution

The dataset used in this research is available through the original Kaggle page:

[Air Quality Dataset — Indian Cities (2022–2025)](https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025/data)

Please refer to the dataset page for its original description, attribution requirements, and licensing information.

---

<p align="center">
  <strong>Understanding when predictive models fail is as important as understanding when they succeed.</strong>
</p>

<p align="center">
  <sub>Research • Experimentation • Robustness • Environmental Data Science</sub>
</p>
