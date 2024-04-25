import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io
import numpy as np

# Upload the CSV file
uploaded = files.upload()
filename = next(iter(uploaded))

# Load the dataset
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Display the column names to check for accuracy
print(df.columns)

# Update the column names if necessary (this step assumes 'Area (km²)' is the correct column name)
features_for_clustering = ['2022 Population', 'Area (km²)']

# Check for any non-numeric data in the DataFrame and handle it
df[features_for_clustering] = df[features_for_clustering].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=features_for_clustering, inplace=True)

# Scale the features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_for_clustering])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.4f}')

# Create scatter plot for clustering
sns.scatterplot(data=df, x='2022 Population', y='Area (km²)', hue='Cluster')
plt.title('Scatter Plot of 2022 Population vs Area by Cluster')
plt.show()

# Histogram of population growth rate
sns.histplot(df['Growth Rate'].dropna(), kde=True)
plt.title('Histogram of Population Growth Rate')
plt.show()

# Line plot for growth rate over countries
sns.lineplot(data=df, x='Country/Territory', y='Growth Rate', estimator='mean', ci=None)
plt.title('Growth Rate by Country')
plt.xticks(rotation=90)  # Rotate the x labels for better readability
plt.show()

# Linear regression for 'Density (per km²)' based on '2022 Population'
# Ensure 'Density (per km²)' is the correct column name
df['Density (per km²)'] = pd.to_numeric(df['Density (per km²)'].replace('*', np.nan), errors='coerce')
df.dropna(subset=['Density (per km²)'], inplace=True)

X = df[['2022 Population']]
y = df['Density (per km²)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Plotting regression line over the scatter plot
plt.scatter(X, y, color='blue')
plt.plot(X, regressor.predict(X), color='red', linewidth=2)
plt.title('Regression Line for Density (per km²)')
plt.xlabel('2022 Population')
plt.ylabel('Density (per km²)')
plt.show()
