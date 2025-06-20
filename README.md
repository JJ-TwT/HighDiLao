# HighDiLao
Built for the Xylem Global Student Innovation Challenge 2025 💧

HighDiLao is a synthetic weather data generator and flood detection system. It is a full-stack climate data simulation pipeline that:
* Generates synthetic daily precipitation data using Markov Chains and Kernel Density Estimation (KDE) based on historical data provided
* Trains an XGBoost classifier on historical rainfall patterns to detect heavy storms and flag flash flood warnings
* Visualizes predicted storm periods and dangerous rainfall spikes over a synthetic timeline

Our goal is to simulate realistic rainfall data and predict flood-prone periods, allowing communities and governments to better prepare for extreme weather events, especially in urban areas like Singapore, hence building flood resilience in our community. HighDiLao would also help to increase the accuracy of current forecasting models by providing accurate synthetic data to train these models.

### Set up
* Libraries
  ```sh
  pip install meteostat
  pip install pandas
  pip install matplotlib
  pip install numpy
  pip install random
  pip install scikit-learn
  pip install xgboost
  ```
### What it does
1. Historical Weather Data Retrieval
Uses the Meteostat API to pull daily precipitation data for Singapore (2021–2025).

2. Precipitation State Classification
Each day is classified into:

D – Dry (rainfall < 1mm)

L – Light (1mm ≤ rainfall < 10mm)

H – Heavy (rainfall ≥ 10mm)

3. Monthly Markov Chain Model
Calculates monthly transition matrices between D/L/H states to simulate realistic weather state transitions over time.

4. Kernel Density Estimation (KDE)
For each month, KDE is used to generate a plausible precipitation value matching its classified state (D/L/H).

5. XGBoost Classifier
An XGBoost model is trained on labeled historical flood periods to:

Learn rainfall sequences that precede heavy storms

Identify similar sequences in synthetic data

6. Visualization
Displays:

Blue line: Daily rainfall

Orange dots: Flash flood warnings (rainfall > 20mm)

Green shaded areas: Predicted heavy storm periods
### Features

### How we built it

### What’s next for HighDiLao

### Acknowledgements


