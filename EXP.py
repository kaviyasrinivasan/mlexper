# PCA
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[17,7,16,19],
       [12,5,9,21]])
X_transposed = X.T
print("Step 2: Transposed Data:")
print(X_transposed)
X_mean = np.mean(X_transposed, axis=0)
X_centered = X_transposed - X_mean
print("\nStep 3: Centered Data (Subtract Mean):")
print(X_centered)
cov_matrix = np.cov(X_centered, rowvar=False)
print("\nStep 4: Covariance Matrix:")
print(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nStep 5: Eigenvalues:")
print(eigenvalues)
print("\nStep 5: Eigenvectors:")
print(eigenvectors)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
print("\nStep 6: Sorted Eigenvalues:")
print(sorted_eigenvalues)
print("\nStep 6: Sorted Eigenvectors:")
print(sorted_eigenvectors)
X_pca_manual = np.dot(X_centered, sorted_eigenvectors)
print("\nStep 7: Projected Data (Manual PCA):")
print(X_pca_manual)
plt.figure(figsize=(8, 6))
plt.scatter(X_transposed[:, 0], X_transposed[:, 1], color='blue', label='Original Data')
plt.scatter(X_centered[:, 0], X_centered[:, 1], color='red', label='Centered Data')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA Transformation with X and Y Axes')
plt.legend()
plt.grid(True)
plt.show()



# SVD


import numpy as np
A=np.array([[3,0],[4,5]])
U,sigma,VT=np.linalg.svd(A)
sigma_matrix = np.diag(sigma)
print("Matrix A in SVD form: ")
print("U Matrix: ")
print(U)
print("\n Sigma Matrix: ")
print(sigma_matrix)
print("\nVT Matrix: ")
print(VT)
print("\n A=[U][sigma_matric][VT] form:")
print("\n[U]")
print(U)
print("\n[sigma]")
print(sigma_matrix)
print("\n[VT]")
print(VT)


# LDA
import numpy as np
import matplotlib.pyplot as plt
X1 = np.array([[4, 2], [2, 4], [2, 3], [3, 6], [4, 4]])
X2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
mean1 = np.mean(X1, axis=0)
mean2 = np.mean(X2, axis=0)
print("Mean vector for class 1:", mean1)
print("Mean vector for class 2:", mean2)
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)
Sw = S1 + S2
print("\nS1 (Covariance matrix for class 1):\n", S1)
print("\nS2 (Covariance matrix for class 2):\n", S2)
print("\nSw (Within-class scatter matrix):\n", Sw)
mean_diff = (mean1 - mean2).reshape(2, 1)
Sb = mean_diff @ mean_diff.T
print("\nSb (Between-class scatter matrix):\n", Sb)
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
W = eigenvectors[:, np.argmax(eigenvalues)]
W_normalized = W / W[0]
print("\nNormalized projection vector W:", W_normalized)
Y1 = X1 @ W_normalized
Y2 = X2 @ W_normalized
print("\nProjected data for class 1:", Y1)
print("Projected data for class 2:", Y2)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X1[:, 0], X1[:, 1], label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], label='Class 2')
plt.title('Before applying LDA')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.subplot(122)
plt.scatter(Y1, np.zeros_like(Y1), label='Class 1')
plt.scatter(Y2, np.zeros_like(Y2), label='Class 2')
plt.title('After applying LDA')
plt.xlabel('Projection axis')
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()

# EXP3 KNN

from math import sqrt
from collections import Counter
import pandas as pd
data = [
  [5.3, 3.7, 'Setosa'],
  [5.1, 3.8, 'Setosa'],
  [7.2, 3.0, 'Virginica'],
  [5.4, 3.4, 'Setosa'],
  [5.1, 3.3, 'Setosa'],
  [5.4, 3.9, 'Setosa'],
  [7.4, 2.8, 'Virginica'],
  [6.1, 2.8, 'Versicolor'],
  [7.3, 2.9, 'Virginica'],
  [6.0, 2.7, 'Versicolor'],
  [5.8, 2.8, 'Virginica'],
  [6.3, 2.3, 'Versicolor'],
  [5.1, 2.5, 'Versicolor'],
  [6.3, 2.5, 'Versicolor'],
  [5.4, 2.4, 'Versicolor']
]
new_instance = [5.2, 3.1]

def euclidean_distance(point1, point2):
  return sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

for instance in data:
  distance = euclidean_distance(new_instance, instance[:2])
  instance.append(distance)
df = pd.DataFrame(data, columns=['Sepal Length', 'Sepal Width', 'Species', 'Distance'])
df['Rank'] = df['Distance'].rank(method='min').astype(int)
print("Table with Distance and Rank:")
print(df)

def get_neighbors(df, k):
  df_sorted = df.sort_values('Distance').head(k)
  return df_sorted


def predict_classification(neighbors):
  classes = neighbors['Species'].values
  majority_vote = Counter(classes).most_common(1)[0][0]
  return majority_vote

for k in [1, 2, 3]:
  neighbors = get_neighbors(df, k)
  predicted_class = predict_classification(neighbors)
  print(f"\nPredicted class for k={k}: {predicted_class}")


# Naive Bayes
import pandas as pd
from collections import defaultdict
data = {
  'Weather Condition': ['Rainy', 'Rainy', 'OverCast', 'Sunny', 'Sunny', 'Sunny', 'OverCast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'OverCast', 'OverCast', 'Sunny'],
  'Wins in last 3 matches': ['3 wins', '3 wins', '3 wins', '2 wins', '1 win', '1 win', '1 win', '2 wins', '1 win', '2 wins', '2 wins', '2 wins', '3 wins', '2 wins'],
  'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
  'Win toss': ['FALSE', 'TRUE', 'FALSE', 'FALSE', 'FALSE', 'TRUE', 'TRUE', 'FALSE', 'FALSE', 'FALSE', 'TRUE', 'TRUE', 'FALSE', 'TRUE'],
  'Won match?': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Training Data:")
print(df)

def calculate_frequency_table(df):
  freq_table = defaultdict(lambda: defaultdict(int))
  total_count = defaultdict(int)
  for _, row in df.iterrows():
    label = row['Won match?']
    total_count[label] += 1
    for column in df.columns[:-1]:
      freq_table[column][(row[column], label)] += 1
  return freq_table, total_count
freq_table, total_count = calculate_frequency_table(df)
print("\nFrequency Table:")
for feature, values in freq_table.items():
  print(f"\nFeature: {feature}")
  for (value, label), count in values.items():
    print(f" Value: {value}, Label: {label} => Count: {count}")

def calculate_probabilities(freq_table, total_count):
  probabilities = defaultdict(lambda: defaultdict(float))
  for feature in freq_table:
    for (value, label), count in freq_table[feature].items():
      probabilities[label][(feature, value)] = count / total_count[label]
  return probabilities
probabilities = calculate_probabilities(freq_table, total_count)
print("\nCumulative Probabilities:")

for label, values in probabilities.items():
  print(f"\nLabel: {label}")
  for (feature, value), prob in values.items():
    print(f" Feature: {feature}, Value: {value} => Probability: {prob}")

def predict(test_data, probabilities, total_count, alpha=1):
  labels = total_count.keys()
  label_probs = {}
  for label in labels:
    prob = 1 
    for feature, value in test_data.items():
      feature_prob = probabilities[label].get((feature, value), alpha / (total_count[label] + alpha * len(probabilities[label])))
      prob *= feature_prob
    label_probs[label] = prob
  total_prob = sum(label_probs.values())
  if total_prob > 0:
    for label in label_probs:
      label_probs[label] /= total_prob 
  swapped_probs = {
    "Yes": label_probs.get("No", 0),
    "No": label_probs.get("Yes", 0)
  }
  print(f"\nNormalized Probabilities: {swapped_probs}")
  return swapped_probs

test_data = {
  'Weather Condition': 'Rainy',
  'Wins in last 3 matches': '2 wins',
  'Humidity': 'Normal',
  'Win toss': 'TRUE'
}


probabilities_result = predict(test_data, probabilities, total_count)
print(f'\nNormalized Probabilities for the test data: {probabilities_result}')
predicted_class = max(probabilities_result, key=probabilities_result.get)
print(f'\nThe predicted class for the test data is: {predicted_class}')



# Decision Tree

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


data = {
  'Age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
  'Income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
  'Student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
  'Credit_Rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
  'Buys_Computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)
def encode_column(column, mapping=None):
  if mapping is None:
    unique_values = column.unique()
    mapping = {value: i for i, value in enumerate(unique_values)}
  return column.map(mapping), mapping

encoded_data = {}
mappings = {}
for column in df.columns:
  encoded_data[column], mappings[column] = encode_column(df[column])
encoded_df = pd.DataFrame(encoded_data)
X = encoded_df.drop('Buys_Computer', axis=1)
y = encoded_df['Buys_Computer']
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)
new_example_data = {
  'Age': '<=30',
  'Income': 'medium',
  'Student': 'yes',
  'Credit_Rating': 'fair'
}

new_example_encoded = {}
for feature, value in new_example_data.items():
  if value in mappings[feature]:
    new_example_encoded[feature] = mappings[feature][value]
  else:
    new_value = max(mappings[feature].values()) + 1
    mappings[feature][value] = new_value
    new_example_encoded[feature] = new_value
new_example = pd.DataFrame([new_example_encoded])
prediction = clf.predict(new_example)
print(f"Prediction for the new example: {'Buys computer' if prediction[0] == 1 else 'Does not buy computer'}")
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()


# Random Forest

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


data = {
  'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
  'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
  'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
  'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent', 'fair'],
  'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
table = pd.DataFrame(data, columns=["age", "income", "student", "credit_rating", "buys_computer"])
encoder = LabelEncoder()
for column in table:
  table[column] = encoder.fit_transform(table[column])
X = table.iloc[:, 0:4].values # Features (first 4 columns)
y = table.iloc[:, 4].values  # Target (last column)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
model = RandomForestClassifier(n_estimators=3, random_state=2)
model.fit(X_train, y_train)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
for idx, tree in enumerate(model.estimators_):
  plot_tree(tree, feature_names=table.columns[:4], class_names=['No', 'Yes'], filled=True, ax=axes[idx])
  axes[idx].set_title(f'Decision Tree {idx+1}')
plt.tight_layout()
plt.show()
test_data = [[0, 2, 1, 0]]
prediction = model.predict(test_data)
if prediction == 1:
  print("Prediction: Buys Computer")
else:
  print("Prediction: Doesn't Buy Computer")



#EXP5 SVM
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
X = np.array([[4, 1], [4, -1], [6, 0], [1, 0], [0, 1], [0, -1]])
y = np.array([1, 1, 1, -1, -1, -1]) # 1 for positive class, -1 for negative class
clf = svm.SVC(kernel='linear', C=1000)
def plot_graph(ax, X, y, title, draw_hyperplane=False):
  ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
  ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
  ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, label='Positive Class')
  ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', s=50, label='Negative Class')
  for i, (x, y_coord) in enumerate(X):
    ax.annotate(f'({x},{y_coord})', (x, y_coord), xytext=(5, 5), textcoords='offset points')
  if draw_hyperplane:
    w = clf.coef_[0]
    b = clf.intercept_[0]
    x_points = np.array([-1, 7])
    y_points = -(w[0] * x_points + b) / w[1]
    ax.plot(x_points, y_points, 'g-', label='Hyperplane')
  ax.set_xlim(-3, 7)
  ax.set_ylim(-3, 4)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title(title)
  ax.legend()
  ax.grid(True)
  for i in range(-3, 8):
    ax.text(i, -0.2, str(i), ha='center', va='center')
    if i != 0:
      ax.text(-0.2, i, str(i), ha='center', va='center')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plot_graph(ax1, X, y, "Before applying SVM")
clf.fit(X, y)
plot_graph(ax2, X, y, "After applying SVM", draw_hyperplane=True)
plt.tight_layout()
plt.show()
w = clf.coef_[0]
b = clf.intercept_[0]
print(f"Hyperplane equation: {w[0]:.2f}x + {w[1]:.2f}y + {b:.2f} = 0")
print("Support vectors:")
print(clf.support_vectors_)
test_point = np.array([[5, 1]])
predicted_class = clf.predict(test_point)
class_label = 'Positive' if predicted_class == 1 else 'Negative'
print(f"The test point (5, 1) is predicted to be in the {class_label} class.")



# EXP6 Linear Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
weeks = [1, 2, 3, 4,5]
sales = [1.2, 1.8, 2.6, 3.2,3.8]
plt.figure(figsize=(10, 6))
plt.scatter(weeks, sales, color='red', marker='o')
X = np.array(weeks).reshape(-1, 1)
y = np.array(sales)
reg = LinearRegression().fit(X, y)
intercept = reg.intercept_
slope = reg.coef_[0]
line_x = np.array([0, 5])
line_y = intercept + slope * line_x
plt.plot(line_x, line_y, color='blue')
plt.title('Sales Regression Analysis')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.text(0.1, 0.2, f'Intercept: {intercept:.2f}', transform=plt.gca().transAxes)
plt.text(0.1, 0.1, f'y = {intercept:.2f} + {slope:.2f}x', transform=plt.gca().transAxes)
for i, (x, y) in enumerate(zip(weeks, sales)):
  plt.annotate(f'({x}, {y:.1f})', (x, y), xytext=(5, 5), textcoords='offset points')
month_7 = intercept + slope * 7
month_9 = intercept + slope * 9
print(f"7th month sales: y = {intercept:.2f} + ({slope:.2f} * 7) = {month_7:.2f}")
print(f"9th month sales: y = {intercept:.2f} + ({slope:.2f} * 9) = {month_9:.2f}")
plt.show()
print(f"\nRegression Equation: y = {intercept:.2f} + {slope:.2f}x")
print(f"Intercept: {intercept:.2f}")
print(f"Slope: {slope:.2f}")


# Multiple linear regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = np.array([
  [1, 4],
  [2, 5],
  [3, 8],
  [4, 2]
])

y = np.array([1, 6, 8, 12])
model = LinearRegression()
model.fit(X, y)
print("Coefficients:")
print(f"a0 (Intercept): {model.intercept_:.3f}")
print(f"a1 (Product 1): {model.coef_[0]:.3f}")
print(f"a2 (Product 2): {model.coef_[1]:.3f}")
equation = f"y = {model.intercept_:.3f} + {model.coef_[0]:.3f}x1 + {model.coef_[1]:.3f}x2"
print(f"\nMultiple Linear Regression Equation:\n{equation}")
week_5_data = np.array([[5, 6]]) 
predicted_sales = model.predict(week_5_data)
print(f"\nPredicted 5th week sales: {predicted_sales[0]:.3f} lakhs")
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
ax.set_xlabel('Product 1 Sales')
ax.set_ylabel('Product 2 Sales')
ax.set_zlabel('Weekly Sales (in lakhs)')
ax.set_title('3D Scatter Plot of Sales Data')
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
ax.plot_surface(X1, X2, Z, alpha=0.5)
ax2 = fig.add_subplot(122)
y_pred = model.predict(X)
ax2.scatter(y, y_pred, c='b', marker='o')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Sales')
ax2.set_ylabel('Predicted Sales')
ax2.set_title('Actual vs Predicted Sales')
plt.tight_layout()
plt.show()



# Multiple Linear regression

import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([
  [1, 4],[2, 5],[3, 8],[4, 2]
])
y = np.array([1, 6, 8, 12])
model = LinearRegression()
model.fit(X, y)
a0 = model.intercept_
a1, a2 = model.coef_
print(f"The multiple linear regression equation is,")
print(f"y = {a0:.2f} + {a1:.3f} X1 - {abs(a2):.3f} X2")
x1_5 = 5 
x2_5 = 6 
y_pred = a0 + (a1 * x1_5) + (a2 * x2_5)
print(f"\nThe 5th week sales is predicted as,")
print(f"y = {a0:.2f} + ({a1:.3f} * 5) - ({abs(a2):.3f} * 6)")
print(f"y = {y_pred:.3f} Lakhs")


# Logistic

import numpy as np
import matplotlib.pyplot as plt
def logistic_function(x, a0, a1):
  z = a0 + a1 * x
  return 1 / (1 + np.exp(-z))
a0 = 1
a1 = 8
threshold = 0.5
x = 60
p = a0 + a1 * x
y = logistic_function(x, a0, a1)
selected = y > threshold
print(f"The equation for Logistic regression is:")
print(f"y = 1 / (1 + e^(-x))")
print(f"\nThe probability for x is:")
print(f"p(x) = z = a0 + a1*x")
print(f"\nGiven a0 = {a0}, a1 = {a1}, x = {x} marks, threshold > {threshold}")
print(f"\np(x) = z = {a0} + {a1} * {x} = {p}")
print(f"\nThe logistic regression equation is:")
print(f"y = 1 / (1 + e^(-{p:.2f})) = {y:.10f}")
print(f"\nSince {y:.10f} > {threshold}, the student with marks = {x}, is {'selected' if selected else 'not selected'}")
x_range = np.linspace(0, 100, 1000)
y_range = logistic_function(x_range, a0, a1)
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='Logistic Curve')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axvline(x=x, color='g', linestyle='--', label='Student Score')
plt.scatter(x, y, color='b', s=100, zorder=5, label='Student')
plt.xlabel('Score')
plt.ylabel('Probability of Selection')
plt.title('Logistic Regression for Student Selection')
plt.legend()
plt.grid(True)
plt.show()



# Polynomial 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 4, 9, 15])
def polynomial_regression(X, y, degree=2):
  poly_features = PolynomialFeatures(degree=degree, include_bias=False)
  X_poly = poly_features.fit_transform(X)
  model = LinearRegression()
  model.fit(X_poly, y)
  return model, poly_features
def plot_polynomial_regression(X, y, model, poly_features):
  X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
  X_plot_poly = poly_features.transform(X_plot)
  y_plot = model.predict(X_plot_poly)
  plt.figure(figsize=(10, 6))
  plt.scatter(X, y, color='blue', label='Data points')
  plt.plot(X_plot, y_plot, color='red', label='Polynomial regression')
  plt.xlabel('X')
  plt.ylabel('y')
  plt.title(f'Polynomial Regression (Degree {poly_features.degree})')
  coef = model.coef_.flatten() # Flatten the coefficient array
  eq = f'y = {model.intercept_:.2f}'
  for i, c in enumerate(coef):
    if i == 0:
      eq += f' + {c:.2f}x'
    else:
      eq += f' + {c:.2f}x^{i+1}'
  plt.legend()
  plt.grid(True)
  plt.show()
model, poly_features = polynomial_regression(X, y)
plot_polynomial_regression(X, y, model, poly_features)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
coef = model.coef_.flatten() 
eq = f'y = {model.intercept_:.2f}'
for i, c in enumerate(coef):
  if i == 0:
    eq += f' + {c:.2f}x'
  else:
    eq += f' + {c:.2f}x^{i+1}'
print("\nPolynomial Regression Equation:")
print(eq)


# EXP7 K means

import numpy as np
from sklearn.cluster import KMeans
data = np.array([[1,1], [2,1], [2,3], [3,2], [4,3], [5,5]])
initial_centroids = np.array([[2, 1], [2, 3]])
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, random_state=0).fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
cluster_1 = data[labels == 0]
cluster_2 = data[labels == 1]
print(f"C1 : {cluster_1.tolist()}")
print(f"C2 : {cluster_2.tolist()}")
print(f"Cluster centers : {centers}")


# K mediod
#pip install scikit-learn-extra

import numpy as np
from sklearn_extra.cluster import KMedoids
data = {'x': [7, 2, 3, 8, 7, 4, 6, 7, 6, 3],
    'y': [6, 6, 8, 5, 4, 7, 2, 3, 4, 4]}
x = [[i, j] for i, j in zip(data['x'], data['y'])]
data_x = np.asarray(x)
initial_medoids = np.array([[3, 4], [7, 4]]) # Example: medoids are (7, 6) and (4, 7)
model_km = KMedoids(n_clusters=2, init=initial_medoids, random_state=0)
km = model_km.fit(data_x)
labels = km.labels_
cluster_1 = [x[i] for i in range(len(labels)) if labels[i] == 0]
cluster_2 = [x[i] for i in range(len(labels)) if labels[i] == 1]
print("C1:", cluster_1)
print("C2:", cluster_2)
medoids = data_x[km.medoid_indices_]
print("Cluster medoids:", medoids)


# Hierarchical

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
distance_matrix = np.array([
[0, 9, 3, 6, 11],
[9, 0, 7, 5, 10],
[3, 7, 0, 9, 2],
[6, 5, 9, 0, 8],
[11, 10, 2, 8, 0]
])
condensed_distance_matrix = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
Z = linkage(condensed_distance_matrix, method='complete')
plt.figure(figsize=(8, 6))
dendrogram(Z, labels=['a', 'b', 'c', 'd', 'e'], leaf_rotation=90)
plt.title("Dendrogram for Complete-Linkage Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
X = linkage(condensed_distance_matrix, method='average')
plt.figure(figsize=(8, 6))
dendrogram(X, labels=['a', 'b', 'c', 'd', 'e'], leaf_rotation=90)
plt.title("Dendrogram for Average-Linkage Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
Y = linkage(condensed_distance_matrix, method='single')
plt.figure(figsize=(8, 6))
dendrogram(Y, labels=['a', 'b', 'c', 'd', 'e'], leaf_rotation=90)
plt.title("Dendrogram for single-Linkage Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()



# DB Scan

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict

data = np.array([
  [3, 7], [4, 6], [5, 5], [6, 4], [7, 3],
  [6, 2], [7, 2], [8, 4], [3, 3], [2, 6],
  [3, 5], [2, 4]
])

eps = 1.9 
min_samples = 4 
dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
labels = dbscan.labels_
core_sample_indices = dbscan.core_sample_indices_
core_points = set(core_sample_indices)
border_points = set()
noise_points = set()
for i, label in enumerate(labels):
  if i not in core_points:
    if label != -1:
      border_points.add(i)
    else:
      noise_points.add(i)
connections = defaultdict(list)
for i in range(len(data)):
  for j in range(i + 1, len(data)):
    if np.linalg.norm(data[i] - data[j]) <= eps:
      connections[i].append(j)
      connections[j].append(i)
plt.figure(figsize=(12, 8))
colors = ['red' if i in core_points else 'blue' if i in border_points else 'gray' for i in range(len(data))]
plt.scatter(data[:, 0], data[:, 1], c=colors, s=100)
for i, (x, y) in enumerate(data):
  plt.annotate(f'P{i + 1}', (x, y), xytext=(5, 5), textcoords='offset points')
for i, connected in connections.items():
  for j in connected:
    plt.plot([data[i][0], data[j][0]], [data[i][1], data[j][1]], 'k-', alpha=0.3)
plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
print("Point Classifications:")
for i in range(len(data)):
  if i in core_points:
    print(f"P{i + 1}: Core")
  elif i in border_points:
    core_neighbors = [f"P{j + 1}" for j in connections[i] if j in core_points]
    print(f"P{i + 1}: Border (Part of Core {', '.join(core_neighbors)})")
  else:
    print(f"P{i + 1}: Noise (Not a part of any Core)")
print("\nConnections:")
for i, connected in connections.items():
  print(f"P{i + 1}: {', '.join([f'P{j + 1}' for j in connected])}")



#EXP8 Neural network
import numpy as np

class NeuronNetwork:
    def __init__(self):
        self.weights = np.array([0.1, 0.3, -0.2])
        self.inputs = np.array([0.8, 0.6, 0.4])
        self.bias = 0.35

    def binary_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def bipolar_sigmoid(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def identity(self, x):
        return x

    def threshold(self, x):
        return 1 if x >= 0 else 0

    def relu(self, x):
        return max(0, x)

    def hyperbolic_tangent(self, x):
        return np.tanh(x)

    def calculate_net_input(self):
        return self.bias + np.sum(self.inputs * self.weights)

    def compute_outputs(self):
        net_input = self.calculate_net_input()
        return {
            'net_input': net_input,
            'binary_sigmoid': self.binary_sigmoid(net_input),
            'bipolar_sigmoid': self.bipolar_sigmoid(net_input),
            'identity': self.identity(net_input),
            'threshold': self.threshold(net_input),
            'relu': self.relu(net_input),
            'hyperbolic_tangent': self.hyperbolic_tangent(net_input)
        }

network = NeuronNetwork()
results = network.compute_outputs()

print(f"Net input (y_in): {results['net_input']:.3f}")
print(f"Binary sigmoid output: {results['binary_sigmoid']:.3f}")
print(f"Bipolar sigmoid output: {results['bipolar_sigmoid']:.3f}")
print(f"Identity output: {results['identity']:.3f}")
print(f"Threshold output: {results['threshold']}")
print(f"ReLU output: {results['relu']:.3f}")
print(f"Hyperbolic tangent output: {results['hyperbolic_tangent']:.3f}")


#EXP9 Reinforcement

import numpy as np
rows, cols = 3, 4
values = np.zeros((rows, cols))
rewards = np.zeros((rows, cols))
rewards[0, 3] = 1
rewards[1, 3] = -10
gamma = 0.9
actions = ['up', 'down', 'left', 'right']
action_moves = {
  'up': (-1, 0),
  'down': (1, 0),
  'left': (0, -1),
  'right': (0, 1)
}
state_labels = {index: f's{index + 1}' for index in range(rows * cols)}
def value_iteration(values, rewards, gamma, threshold=1e-4):
  rows, cols = values.shape
  delta = float('inf')
  while delta > threshold:
    delta = 0
    for i in range(rows):
      for j in range(cols):
        if (i, j) == (0, 3) or (i, j) == (1, 3):
          continue
        v = values[i, j]
        possible_values = []
        for action in actions:
          di, dj = action_moves[action]
          ni, nj = i + di, j + dj
          if 0 <= ni < rows and 0 <= nj < cols:
            if (ni, nj) == (1, 1):
              continue
            possible_values.append(rewards[ni, nj] + gamma * values[ni, nj])
        if possible_values:
          values[i, j] = max(possible_values)
        delta = max(delta, abs(v - values[i, j]))
  return values
def extract_policy(values, rewards, gamma):
  rows, cols = values.shape
  policy = np.full((rows, cols), None)
  for i in range(rows):
    for j in range(cols):
      if (i, j) == (0, 3):
        policy[i, j] = 'goal'
        continue
      elif (i, j) == (1, 3):
        policy[i, j] = 'fire'
        continue
      elif (i, j) == (1, 1):
        policy[i, j] = 'wall'
        continue
      best_action = None
      best_value = -float('inf')
      for action in actions:
        di, dj = action_moves[action]
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
          if (ni, nj) == (1, 1):
            continue
          action_value = rewards[ni, nj] + gamma * values[ni, nj]
          if action_value > best_value:
            best_value = action_value
            best_action = action
      policy[i, j] = best_action
  return policy
def find_optimal_path(start, policy):
  path = [start]
  current = start
  while policy[current] != 'goal' and policy[current] != 'fire':
    action = policy[current]
    if action is None:
      break
    di, dj = action_moves[action]
    current = (current[0] + di, current[1] + dj)
    path.append(current)
    if current == (0, 3) or current == (1, 3):
      break
  labeled_path = [state_labels[i * cols + j] for i, j in path]
  return labeled_path
values = value_iteration(values, rewards, gamma)
policy = extract_policy(values, rewards, gamma)
print("Values:")
for i in range(rows):
  for j in range(cols):
    state = state_labels[i * cols + j]
    print(f"{state}: {values[i, j]:.2f}", end='\t')
  print()
print("\nPolicy:")
for i in range(rows):
  for j in range(cols):
    state = state_labels[i * cols + j]
    print(f"{state}: {policy[i, j]}", end='\t')
  print()
start_state_input = input("Enter the starting state (e.g., s1, s2, ..., s12): ").strip()
start_state_index = {v: k for k, v in state_labels.items()}.get(start_state_input)
if start_state_index is not None:
  start_row, start_col = divmod(start_state_index, cols)
  start_state = (start_row, start_col)
  optimal_path = find_optimal_path(start_state, policy)
  print("Optimal path:", optimal_path)
else:
  print("Invalid starting position.")



#EXP10 Cross validation

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
data = load_iris()
X, y = data.data, data.target
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", scores)
print("Mean Cross-Validation Accuracy:", np.mean(scores))
loo = LeaveOneOut()
loocv_scores = cross_val_score(knn, X, y, cv=loo, scoring='accuracy')
print("LOOCV Accuracy Scores:", loocv_scores)
print("Mean LOOCV Accuracy:", np.mean(loocv_scores))



# Apriori


from itertools import combinations
def calculate_support(transactions, itemset):
  count = 0
  for transaction in transactions:
    if itemset.issubset(transaction):
      count += 1
  return count / len(transactions)

def generate_candidates(prev_freq_itemsets, k):
  candidates = set()
  prev_freq_itemsets_list = list(prev_freq_itemsets)
  for i in range(len(prev_freq_itemsets_list)):
    for j in range(i + 1, len(prev_freq_itemsets_list)):
      union_set = prev_freq_itemsets_list[i].union(prev_freq_itemsets_list[j])
      if len(union_set) == k:
        candidates.add(union_set)
  return candidates

def prune_candidates(candidates, prev_freq_itemsets):
  pruned_candidates = set()
  for candidate in candidates:
    all_subsets_frequent = True
    for subset in combinations(candidate, len(candidate) - 1):
      if frozenset(subset) not in prev_freq_itemsets:
        all_subsets_frequent = False
        break
    if all_subsets_frequent:
      pruned_candidates.add(candidate)
  return pruned_candidates

def apriori(transactions, min_support):
  itemsets = set()
  for transaction in transactions:
    for item in transaction:
      itemsets.add(frozenset([item]))
  freq_itemsets = {itemset for itemset in itemsets if calculate_support(transactions, itemset) >= min_support}
  all_freq_itemsets = dict()
  all_freq_itemsets[1] = freq_itemsets
  k = 2
  while len(all_freq_itemsets[k - 1]) > 0:
    candidates = generate_candidates(all_freq_itemsets[k - 1], k)
    candidates = prune_candidates(candidates, all_freq_itemsets[k - 1])
    freq_itemsets_k = {candidate for candidate in candidates if calculate_support(transactions, candidate) >= min_support}
    all_freq_itemsets[k] = freq_itemsets_k
    k += 1
  return all_freq_itemsets

def generate_association_rules(freq_itemsets, transactions, min_confidence):
  rules = []
  for k, itemsets in freq_itemsets.items():
    if k >= 2: # We only generate rules for itemsets of size 2 or greater
      for itemset in itemsets:
        for i in range(1, len(itemset)):
          for subset in combinations(itemset, i):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            antecedent_support = calculate_support(transactions, antecedent)
            rule_support = calculate_support(transactions, itemset)
            confidence = rule_support / antecedent_support
            if confidence >= min_confidence:
              rules.append((antecedent, consequent, rule_support, confidence))
  return rules


transactions = [
  frozenset(['butter', 'bread', 'milk']),
  frozenset([ 'bread', 'butter']),
  frozenset(['beer', 'cookies', 'diapers']),
  frozenset(['milk', 'diapers', 'bread', 'butter']),
  frozenset(['beer', 'diapers'])
]
min_support = 0.4
min_confidence = 0.7
freq_itemsets = apriori(transactions, min_support)
rules = generate_association_rules(freq_itemsets, transactions, min_confidence)
print("Frequent Itemsets:")
for k, itemsets in freq_itemsets.items():
  if itemsets:
    print(f"Frequent {k}-itemsets:")
    for itemset in itemsets:
      print(f"{set(itemset)}")
print("\nStrong Association Rules")
for rule in rules:
  antecedent, consequent, support, confidence = rule
  if confidence == 1:
    print(f"{set(antecedent)} -> {set(consequent)}")

