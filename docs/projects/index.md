# Projects

The course project is done in a team of four (Ana Helena Caiafa, Thais Rocha,
Esther Caroline Rodrigues and Carolina Eskenazi) and has two tasks — one
**classification** and one **regression** — on tabular data. It is built in
stages: first an exploratory data analysis (EDA) and a preprocessing pipeline
for each dataset, then the neural-network models themselves.

| Stage | Status |
| --- | --- |
| [Exploratory data analysis](#exploratory-data-analysis) — both datasets | Complete |
| Classification model | Not started |
| Regression model | Not started |

## Exploratory data analysis

The EDA lives in its own team repository and site:

- **Site:** [thaisrs165.github.io/exploratory_data_analysis](https://thaisrs165.github.io/exploratory_data_analysis/)
- **Code:** [github.com/Thaisrs165/exploratory_data_analysis](https://github.com/Thaisrs165/exploratory_data_analysis)

No model is trained there; each report ends with a synthesis for whoever builds
the model. Both reports follow the same stages: proposal, initial inspection,
univariate analysis, bivariate and multivariate analysis, preprocessing
(including PCA, t-SNE and UMAP), and synthesis.

<div class="grid cards" markdown>

-   **Classification — UCI Bank Marketing**

    ---

    Will a client subscribe to a term deposit after a marketing call?

    - 45,211 calls, 16 input features, binary target `y`
    - Imbalanced: 11.70% `yes` (always predicting `no` already gives 88.30% accuracy)
    - `duration` dropped as target leakage
    - Contact history (`poutcome`) is the strongest signal
    - Stratified 80/20 split; 51 features after the pipeline

    [:octicons-arrow-right-24: Read the report](https://thaisrs165.github.io/exploratory_data_analysis/projects/eda/stage-0/)

-   **Regression — UCI Seoul Bike Sharing Demand**

    ---

    How many public bikes are rented in Seoul in a given hour?

    - 8,760 hours, 13 input columns, target `rented_bike_count`
    - Right-skewed target: train on `sqrt(y)`
    - Closed hours (`functioning_day`) removed as target leakage
    - Demand driven non-linearly by temperature and hour; rain and snow switch it off
    - Split by whole days, stratified by season; 22 features after the pipeline

    [:octicons-arrow-right-24: Read the report](https://thaisrs165.github.io/exploratory_data_analysis/projects/eda-regression/stage-0/)

</div>

Both preprocessing pipelines (`code/bank_preprocessing.py` and
`code/bike_preprocessing.py`) are importable `scikit-learn` `Pipeline` +
`ColumnTransformer` objects fitted on the training set only, and will be reused
as-is when the models are trained.

## Classification

_Not started yet._

## Regression

_Not started yet._
