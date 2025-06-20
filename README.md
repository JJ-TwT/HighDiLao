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
### Features
1. 🌧️ Historical Weather Data Retrieval using the Meteostat API to pull daily precipitation data for Singapore (2021–2025).

2. 📝 Precipitation State Classification:
Each day is classified into:
* D – Dry (rainfall < 1mm)
* L – Light (1mm ≤ rainfall < 10mm)
* H – Heavy (rainfall ≥ 10mm)
  
3. 📅 Monthly Markov Chain Model:
Calculates monthly transition matrices between D/L/H states to simulate realistic weather state transitions over time.

4. 🔢 Kernel Density Estimation (KDE):
For each day in the simulation period, KDE is used to generate a plausible precipitation value matching its classified state (D/L/H).

5. 🤖 XGBoost Classifier:
An XGBoost model is trained on labeled historical flood periods fromm local data to:
* Learn rainfall sequences that precede heavy storms and flash floods
* Identify similar sequences in the provided synthetic data

6. 📈 Visualization:
Displays the synthetic rainfall graph of the given simulation period which showcases the predicted heavy storm periods and flash flood warnings

### What’s next for HighDiLao
* 📊 Incorporate more climate variables (humidity, wind speed, temperature etc.)
* 🗺️ Expand beyond Singapore to a more global scale (with access to more weather stations) so as to simulate other flood-prone cities
* 📡 Add real-time alerting or IoT sensor integration
* 💻 Deploy as a web-based  support tool for the public, government and urban planners
* ✅ Benchmark against traditional models to validate accuracy and show that HighDiLao's synthetic data outperforms older methods in precision
* 🛰️ Integrate satellite imagery data with synthetic precipitation to enhance realism and accuracy -> hybrid approaches where GANs (Generative Adversarial Networks) can be used to learn from both image-based and numerical data sources for more powerful climate simulations

### Resources
* https://meteostat.net/en/


