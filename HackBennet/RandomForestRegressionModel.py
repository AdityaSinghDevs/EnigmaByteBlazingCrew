import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('clean.csv')
proj_mapping = {
    'Urban Development' : 1,
    'Energy' : 2,
    'Construction' : 3,
    'Transportation' : 4,
    'Water Management' : 5
}

data['Project_Type'] = data['Project_Type'].replace(proj_mapping)
print(data.head())

features = ['Project_Type','Area_Impacted','Air_Emissions','Water_Pollution','Habitat_Loss','Carbon_Footprint']

x = data[features]
y = data.Impact_Score
print(x.head())

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

model = RandomForestRegressor(max_depth =10, random_state=10, max_leaf_nodes = 8)
model.fit(train_x, train_y)

prediction = model.predict(val_x)
print(mean_absolute_error(val_y, prediction))
plt.plot([min(val_y), max(val_y)], [min(val_y), max(val_y)], color='red', linestyle='--')
plt.xlabel('Orignal')
plt.ylabel('pred')
plt.plot()
print(prediction[:5])
print(data['Impact_Score'].head())

# Make predictions on the validation set
prediction = model.predict(val_x)

# Calculate mean squared error (already included)
mse = mean_squared_error(val_y, prediction)
print("Mean Squared Error:", mse)

# Calculate R-squared (accuracy)
accuracy = r2_score(val_y, prediction)
print("R-squared (Accuracy):", accuracy)