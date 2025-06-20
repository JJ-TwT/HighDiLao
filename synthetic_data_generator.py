from meteostat import Point, Daily
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

#Ask user to input start and end date
#Let the fixed dates be 2021-01-02 to 2025-06-17. this will be all the historical data provided to the program for it to calculate the probabilities for the markov chains
start_input = "2021-01-02" #input("Enter start date (YYYY-MM-DD): ")
end_input = "2025-06-17" #input("Enter end date (YYYY-MM-DD): ")

#Convert input to datetime objects (only needed when users input the start and end date)
try:
    start = datetime.strptime(start_input, "%Y-%m-%d")
    end = datetime.strptime(end_input, "%Y-%m-%d")
except ValueError:
    print("Invalid date format. Please use YYYY-MM-DD.")
    exit()

#Define Singapore location
singapore = Point(1.3667, 103.9167)

#Fetch daily weather data
data = Daily(singapore, start, end).fetch()

#Check if any data exists

if data.empty or 'prcp' not in data.columns:
    print("No precipitation data found for this date range.")
else:
    data = data.reset_index()

    #Classify the precipitation values into their respective D, H, L states
    #if precipitation value <1, classify that day as D (dry)
    #if precipitation value 1<x<10, classify that day as L (low)
    #if precipitation value 10<, classify that day as H (high)
    def classify_rainfall(mm):
        if pd.isna(mm):
            return "No Data"
        elif mm <1:
            return "D"
        elif mm < 10:
            return "L"
        else:
            return "H"

    data = data.reset_index()
    data['rainfall_mm'] = data['prcp'].fillna(0.0)
    data['state'] = data['rainfall_mm'].apply(classify_rainfall)
    data['month'] = data['time'].dt.month

    #Clean and display
    output = data[['time', 'rainfall_mm', 'state', 'month']]
    output.columns = ['date', 'rainfall_mm', 'state', 'month']
    #print(output.to_string(index=False)) 

    #Print precipitation states as single line
    state_list = output['state'].tolist() #LIST OF ALL THE HISTORICAL DATA'S PRECIPITATION STATES IN AN ARRAY
    #state_line = ', '.join(f"'{s}'" for s in state_list)
    #print("\nPrecipitation States:")
    #print(state_list)

    #Print precipitation values as single line
    value_list = output['rainfall_mm'].tolist() #LIST OF ALL THE HISTORICAL DATA'S PRECIPITATION VALUES IN AN ARRAY
    #value_line = ', '.join(f"'{s}'" for s in value_list)
    #print("\nPrecipitation Values:")
    #print(value_list)


#graph that plots all the data points for precipitation values based on the historical data given
'''
#fetch and plot average rainfall for each month
if not data.empty:
    data["Month"] = data["time"].dt.month
    monthly_avg = data.groupby("Month")["prcp"].mean()
    
    plt.figure(figsize=(10,5))
    plt.plot(monthly_avg.index, monthly_avg.values, marker="o", linestyle="-", color="blue", label="Average Precipitation")

    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Average Precipitation (mm)")
    plt.title("Seasonal Precipitation Trends in SG")
    plt.show()
'''

#constructing the markov transition matrix 
#finds the transition probabilities for D -> D, D -> L, D -> H and so on
def calculate_transition_probabilities(data):
    state_order = ['D', 'L', 'H']
    state_index = {state: i for i, state in enumerate(state_order)}

    # Initialize transition matrix
    num_states = len(state_order)
    transition_matrix = np.zeros((num_states, num_states))
    
    # Populate transition matrix
    for i in range(len(data) - 1): #-1 because u cant calculate transition probabilities from only the first state
        current_state = state_index[data[i]]
        next_state = state_index[data[i + 1]]
        transition_matrix[current_state, next_state] += 1 
    
    # Normalize to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    return pd.DataFrame(transition_matrix, index=state_order, columns=state_order)

monthly_matrices = {}

#creates a separate transition matrix for each month to capture seasonal differences
for month in range(1,13):
    month_states = data.loc[data["month"] == month, "state"].tolist()
    if len(month_states) > 1: #ensures enough states to calculate probabilities
        matrix = calculate_transition_probabilities(month_states)
        monthly_matrices[month] = matrix
'''
for month, matrix in monthly_matrices.items(): #for every month(key) and matrix for the month, 
    print(f"\n transition matrix for month {month}") #f formats it so we can embed variable directly into the text
    print(matrix.fillna(0.0)) 
'''

#state the frequency weights for sampling
#determines the relative frequency of D, L, H per month - used to generate realistic synthetic sequences
monthly_weights = (
    output.groupby("month")["state"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)
#print(monthly_weights)

#compartmentalising synthetic data into different month
#users provide the simulation period 
start_date = input("Enter the start date of synthetic data (YYYY-MM-DD) : ")

try:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
except ValueError:
    print("Invalid format. Please use YYYY-MM-DD.")
    exit()

seqlength_input = int(input("Enter number of days of synthetic states needed : "))

#given the start state, the program walks through each day using transition probabilities to generate a likely next state (D, L, H) based on the month
def generate_markov_sequence(monthly_matrices, start_state, num_days, current_date):
    current_state = start_state
    sequence = []

    for i in range(num_days):
        month = current_date.month 
        matrix = monthly_matrices.get(month)
        next_state = np.random.choice(matrix.columns, p=matrix.loc[current_state].values)
        sequence.append((current_date, current_state))
        current_state = next_state 
        current_date += timedelta(days=1)
    return sequence


import random 
month = start_date.month
weights = monthly_weights.get(month) 
random_start_state = random.choices(['D', 'L', 'H'], weights=weights)[0] #make the start state random based on how frequent the states appear

synthetic_states = generate_markov_sequence(monthly_matrices, start_state=random_start_state, num_days=seqlength_input, current_date=start_date)
'''
for day, state in synthetic_states:
    print(f"{day.date()}: {state}")
'''

#for each month, a Kernal Density Estimate is fitted on historical precipitation to sample realistic rainfall amounts matching a target state
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

PLOT_KDE = False

monthly_kdes = {}
for month in range(1,13):
    month_data = data.loc[data["month"] == month, "rainfall_mm"].fillna(0.0)
    if len(month_data) > 1: #ensures enough states to calculate probabilities
        dataset = month_data.to_numpy().reshape(-1, 1) #reshape to use for KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(dataset) #fit KDE
        monthly_kdes[month] = { "kde": kde, "dataset": dataset}
    
    #Generate values across the range for visualization
    x_range = np.linspace(dataset.min()-0.3, dataset.max()+0.3, num=600)
    log_density = kde.score_samples(x_range[:, np.newaxis])

    if PLOT_KDE:
        plt.figure(figsize=(10, 4))
        plt.plot(x_range, np.exp(log_density), color='gray', linewidth=2.5, label="KDE Curve")

        plt.legend()
        plt.xlabel("Precipitation (mm)")
        plt.ylabel("Density")
        plt.title(f"Synthetic Precipitation Values Using KDE for {month}")
        plt.show()

synthetic_results = []
#based on the generated state (D, L, or H), sample only values that satisfy the expected range
for date, state in synthetic_states:
    month = date.month 
    kde_info = monthly_kdes.get(month)
    if not kde_info:
        continue
    kde = kde_info["kde"]

    ppt = None 
    while ppt is None:
        sample = kde.sample(1)[0][0]
        if sample < 0:
            continue 
        if state == 'D' and sample < 1:
            ppt = sample 
        elif state == 'L' and 1 <= sample < 10:
            ppt = sample
        elif state == 'H' and sample >= 10:
            ppt = sample
        
    synthetic_results.append((date, state, round(ppt, 2)))
print(round(ppt,2))


#builds a dataframe for the synthetic dataset 
dfsynthetic = pd.DataFrame(synthetic_results, columns=["Date", "State", "Precipitation(mm)"])
print(dfsynthetic)

#extracts the precipitation-only list for later model input
pptonly = dfsynthetic['Precipitation(mm)'].to_list()
print(pptonly)

#XGBOOST MODEL TO 
# 1) GRAPH OUT ALL THE SYNTHETICALLY GENERATED RAINFALL 
# 2) PREDICT WHERE THE HEAVY STORM PERIODS WILL BE THROUGHOUT THE GENERATED TIME PERIOD 
# 3) MARK OUT THE DATES WHERE FLASH FLOODS ARE MOST LIKELY TO HAPPEN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report
#first, we need to train the xgboost model in order for it to be able to recognise what exactly is a heavy storm by just reading precipitation values

#Define training data for known heavy storm periods with rainfall values
#hardcoded storm periods
heavy_storm_data = [
    ('2025-01-10', '2025-01-18', [13.3, 66.8, 34.0, 24.9, 16.5, 15.5, 7.4, 3.1, 17.1]),
    ('2025-03-04', '2025-03-06', [10.9, 40.7, 14.4]),
    ('2025-03-31', '2025-04-05', [27.9, 8.3, 26.8, 11.3, 8.7, 12.1]),
    ('2025-04-09', '2025-04-13', [15.1, 9.9, 18.4, 27.6, 17.7]),
    ('2025-04-20', '2025-04-23', [13.5, 9.7, 11.1, 29.3]),
    ('2024-10-11', '2024-10-12', [22.7, 25.9]),
    ('2024-11-20', '2024-11-23', [20.5, 18.2, 7.2, 19.0]),
    ('2024-11-26', '2024-11-28', [19.0, 41.2, 13.6]),
]

#slices the sequences into overlapping sliding windows of size 3 days
window_size = 3

#build storm (label=1) and non-storm (label=0) training examples
#trains the XGBoost model on the windows to learn how to spot storm patterns
def generate_windows(data_list, label=1):
    X = []
    y = []
    for rainfall_seq in data_list:
        arr = np.array(rainfall_seq)
        for i in range(len(arr) - window_size + 1):
            X.append(arr[i:i+window_size])
            y.append(label)
    return X, y

positive_windows = []
positive_labels = []
for _, _, rainfall_seq in heavy_storm_data:
    Xp, yp = generate_windows([rainfall_seq], label=1)
    positive_windows.extend(Xp)
    positive_labels.extend(yp)

#hardcoded negative sequences (precipitation values data where there are no heavy storms)
#taken from real life data
negative_sequences = [
    [6.1, 18.0, 3.6, 9.4, 2.8, 2.3, 16.2, 0.4, 0.0],
    [2.3, 1.2, 13.1, 3.3, 7.4, 3.4, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 1.0, 1.4, 0.0, 0.3, 0.0, 1.0, 0.0, 0.6, 1.3, 0.8, 2.8],
    [0.5, 3.4, 2.7, 10.5, 3.9, 1.1, 3.9, 10.0, 3.9, 6.6, 3.9, 6.4, 1.6, 1.4, 1.4, 0.7],
    [8.5, 7.6, 4.6, 2.6, 4.8, 1.1, 9.1, 3.9,],
    [5.8, 6.3, 5.9, 1.1, 4.0, 10.2],
    [0.9, 9.7, 8.4, 3.5, 8.8, 5.2],
    [0.5, 0.2, 0.0, 1.0, 1.4, 0.0, 0.3, 0.0, 1.0, 0.0, 0.6, 1.3, 0.8, 2.8], #febuary of 2025
    [5.5, 6.6, 1.5, 13.1, 0.6, 8.9, 9.6, 2.6, 5.9, 3.9, 7.7, 8.7, 10.9, 0.5, 3.8, 0.4, 17.7, 5.3, 4.4, 1.1, 5.6, 5.6, 3.2, 5.9, 2.1, 0.5, 0.9, 10.3, 0.3, 4.6]
]

negative_windows = []
negative_labels = []
for seq in negative_sequences:
    Xn, yn = generate_windows([seq], label=0)
    negative_windows.extend(Xn)
    negative_labels.extend(yn)

X_train = np.array(positive_windows + negative_windows)
y_train = np.array(positive_labels + negative_labels)

#take the simulation period and start date that the user inputted before
synthetic_dates = pd.date_range(start_date, periods=seqlength_input)

model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

#creates sliding windows on the synthetic data and classifies them
#predicts whether each window is a storm (labelled 1) or not (labelled 0)
X_synth = []
synth_window_dates = []
for i in range(len(pptonly) - window_size + 1):
    X_synth.append(pptonly[i:i+window_size])
    synth_window_dates.append(synthetic_dates[i + window_size - 1])

X_synth = np.array(X_synth)
y_synth_pred = model.predict(X_synth)

#plots out the results on a graph
plt.figure(figsize=(15,5))
#main rainfall line in the simulation period
plt.plot(synthetic_dates, pptonly, label='Synthetic Daily Rainfall (mm)', color='blue')

#Orange dots: flash flood warning days (>20mm rainfall)
for d, r in zip(synthetic_dates, pptonly):
    if r > 20: #changable benchmark
        plt.plot(d, r, 'o', color='orange', label='Flash Flood Warning', markersize=8)

#Green shaded blocks: predicted heavy storm windows in synthetic data
for i, pred in enumerate(y_synth_pred):
    if pred == 1:
        start_pred = synth_window_dates[i] - pd.Timedelta(days=window_size-1)
        end_pred = synth_window_dates[i]
        plt.axvspan(start_pred, end_pred + pd.Timedelta(days=1), color='green', alpha=0.25, label='Predicted Heavy Storm')

# Clean legend to avoid duplicates
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Heavy Storm Detection - Predictions (Green), Flash Flood Warning (>20mm) (Orange dots)')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.show()

train_preds = model.predict(X_train)
#prints out the performance report of the xgboost model and shows
#   - Precision: Out of all predicted storm windows, how many were correct.
#   - Recall: Out of all actual storm windows, how many were detected.
#   - F1: Harmonic average of precision and recall.
#   - Support: Number of samples for each class in the test set.
print(classification_report(y_train, train_preds))
