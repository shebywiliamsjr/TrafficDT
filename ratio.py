import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

file_path = "YOLO_Vehicle_Detection.xlsx" 
df = pd.read_excel(file_path)

df = df.dropna(subset=['Hit_Ratio', 'Miss_Ratio'])

peaks, _ = find_peaks(df['Miss_Ratio'], height=np.mean(df['Miss_Ratio']) + np.std(df['Miss_Ratio']))

plt.figure(figsize=(12, 6))
plt.plot(df['Frame'], df['Hit_Ratio'], label='Hit Ratio', color='green', marker='o', markersize=3, linestyle='-')
plt.plot(df['Frame'], df['Miss_Ratio'], label='Miss Ratio', color='red', marker='x', markersize=3, linestyle='-')

plt.scatter(df['Frame'].iloc[peaks], df['Miss_Ratio'].iloc[peaks], color='blue', label="High False Negative Zone", marker='D', s=50)

plt.xlabel("Frame Number")
plt.ylabel("Ratio")
plt.title("Hit Ratio & Miss Ratio Over Time (With Detection Failure Zones)")
plt.legend()
plt.grid(True)

plt.show()
