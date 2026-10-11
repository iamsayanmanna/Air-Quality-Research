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
- Quantify performance differences between overall test observations and extreme-pollution observations.
- Identify the strengths and limitations of different model architectures.

## 🔬 Research Design

The workflow follows a structured, time-aware experimental approach:

1. **Exploratory Data Analysis** — investigate data quality, distributions, temporal patterns, and relationships between variables.
2. **Data Preparation** — define the prediction target, select features, handle missing target values, encode categorical variables, and scale features.
3. **Chronological Evaluation** — preserve temporal ordering when splitting training and testing data.
4. **Model Development** — train machine learning and deep learning models.
5. **Overall Evaluation** — assess predictions on the chronological test set.
6. **Extreme-Event Evaluation** — evaluate predictions for observations meeting the predefined extreme-pollution threshold.
7. **Comparative Analysis** — compare overall performance, extreme-event errors, and performance degradation across models.

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
| [03 — Deep Learning](notebooks/03_Deep_Learning_Air_Quality.ipynb) | Deep learning architectures, time-series sequences, and robustness analysis |

Open the notebooks above to explore the code, analysis, experiments, and recorded outputs.

## 📊 Dataset

**Source:** [Air Quality Dataset — Indian Cities (2022–2025), Kaggle](https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025/data)

The local dataset used in the experiments is named `INDIA_AQI_COMPLETE_20251126.csv`.

The research workflow uses air-quality observations from multiple Indian cities. The working dataset contains approximately **842,160 rows across 71 columns**, covering 29 cities and the period 2022–2025.

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
| Time-series sequence length | 24 |
| Input features for deep learning | 14 |
| Deep learning model parameter budget | 59,915 parameters per architecture |
| Maximum training epochs | 20 |
| Batch size | 512 |
| Early stopping patience | 5 epochs |
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |

The deep learning experiments compare four architectures under a matched parameter budget:

- **MLP** — Multilayer Perceptron
- **RNN** — Recurrent Neural Network
- **GRU** — Gated Recurrent Unit
- **Transformer** — Attention-based sequence model

A matched parameter count provides a more controlled comparison of model size, although it does not make the architectures identical in computational cost, inductive bias, or effective learning capacity.

## 📏 Evaluation Metrics

The project evaluates regression performance using:

- **MAE — Mean Absolute Error:** average absolute prediction error.
- **RMSE — Root Mean Squared Error:** penalizes large prediction errors more heavily.
- **R² — Coefficient of Determination:** measures performance relative to a mean-based baseline; negative values indicate performance worse than that baseline on the evaluated sample.

The evaluation is conducted separately for the full chronological test set and the extreme-pollution subset.

## 🧠 Deep Learning Results

The following table summarizes the recorded results from the current deep learning experiment.

| Model | Overall MAE | Overall RMSE | Overall R² | Extreme MAE | Extreme RMSE | Extreme R² |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7.5443 | 17.0179 | 0.9771 | 62.6798 | 88.8446 | 0.9593 |
| RNN | 26.7387 | 94.5771 | 0.2926 | 384.5475 | 573.4059 | -0.6933 |
| GRU | 8.6067 | 52.6969 | 0.7804 | 163.2418 | 325.3639 | 0.4548 |
| Transformer | 22.6433 | 120.1631 | -0.1419 | 556.9644 | 736.0073 | -1.7899 |

*These are experiment-specific results, not a claim that the same ranking will hold for other datasets, splits, or configurations.*

### Key Observations

- The MLP achieved the strongest overall results in this experiment and the lowest errors among the four evaluated architectures.
- The GRU achieved positive R² values on both the overall test set and the extreme-pollution subset, but its errors were higher than those of the MLP.
- The RNN showed substantially higher errors during extreme-pollution observations and a negative extreme-subset R².
- The Transformer recorded the highest extreme-subset errors and a negative R² on both the overall and extreme subsets.
- The results demonstrate that architecture choice can substantially affect prediction behaviour under this experimental setup.

**Important research distinction:** An extreme-subset error does not by itself prove that a model degrades relative to ordinary pollution conditions. A rigorous degradation analysis should compare the extreme subset with a clearly defined non-extreme subset and account for differences in target ranges and sample sizes.

## 🌫️ Why Extreme Pollution Matters

Extreme pollution observations may be relatively uncommon, yet prediction errors during these episodes can be especially consequential.

This research therefore considers more than average model performance. It asks whether models can generalize to unusually high pollution levels and whether their errors remain acceptable in the conditions that are the primary focus of the study.

The objective is to encourage more careful evaluation of predictive systems intended for environmental monitoring and pollution-related decision support.

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
- Equal parameter counts do not guarantee equal training cost or equivalent model capacity.
- Predictive performance alone does not establish causal relationships between pollutants, weather, and air quality.
- These experiments are research results and should not be treated as a substitute for official air-quality monitoring or public-health guidance.

## 🔭 Future Work

- Compare extreme-pollution performance with non-extreme observations using consistent metrics.
- Investigate performance across cities, seasons, and pollution regimes.
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
