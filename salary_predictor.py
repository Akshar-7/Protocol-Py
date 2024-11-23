import csv
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

# Create a Dictionary for storing employee data 
data = {'years_of_experience':[], 'salary':[]}

# Adding employee data to the Dictionary
with open("dataset.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        data['years_of_experience'].append(float(row[1]))
        data['salary'].append(float(row[2]))

# Create a DataFrame with employee data 
df = pd.DataFrame(data) 

# Split the data into features (Years of Experience) and target (Salary) 
x = df[['years_of_experience']]
y = df['salary']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42) 

# Create a Linear Regression model 
model = LinearRegression() 

# Train the model on the training data 
model.fit(x_train, y_train) 

# Make predictions on the test data 
y_pred = model.predict(x_test) 
r2 = r2_score(y_test, y_pred) 

# Calculate the Mean Squared Error (MSE) 
mse = mean_squared_error(y_test, y_pred) 

# Plot the data points and the fitted line 
plt.scatter(x_test, y_test, label='Test Data') 
plt.plot(x_test, y_pred, color='red', label='Fitted Line') 
plt.xlabel('Years of Experience') 
plt.ylabel('Salary') 
plt.title('Linear Regression') 
plt.legend() 
plt.show() 
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared Error: {r2:.2f}")

# Taking Custom Input from User
experience = int(input("Enter the number of years of experience: "))

# Assume you have a new experience value for prediction, for example, 7 years
new_experience = pd.DataFrame({'years_of_experience':[experience]})

# Make predictions using the trained model
predicted_salary = model.predict(new_experience)

# Display the prediction
print(f"Predicted Salary for {new_experience['years_of_experience'][0]} years of experience: {predicted_salary[0]:,.2f}")
