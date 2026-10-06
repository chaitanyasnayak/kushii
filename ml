1)Implementation of Apriori algorithm
from itertools import combinations
T = [
 {'Milk','Bread','Butter'}, {'Bread','Butter'}, {'Milk','Bread'},
 {'Milk','Butter'}, {'Bread','Butter','Eggs'}, {'Milk','Bread','Butter'}
]
S, C = .5, .7
def sup(x): return sum(x <= t for t in T)/len(T)
items = set().union(*T)
freq = []

for k in range(1, len(items)+1):
    f = {frozenset(x) for x in combinations(items,k) if sup(set(x)) >= S}
    if not f: break
    freq += f
print("Frequent Itemsets:")
for x in freq: print(set(x), round(sup(set(x)),2))
print("\nAssociation Rules:")
for x in freq:
    for r in range(1,len(x)):
        for a in combinations(x,r):
            a,b=set(a),set(x)-set(a)
            conf=sup(a|b)/sup(a)
            if conf>=C: print(a,"->",b,round(conf,2))



2)Implementation of Classification mode K-nearest neighbour
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = [[1,2], [2,3], [3,4], [6,7], [7,8], [8,9]]
y = ['Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", pred)
print("Accuracy:", accuracy_score(y_test, pred))

print("New Prediction:", model.predict([[5,6]]))




3)Implementation of linear regression model.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dataset
X = [[1], [2], [3], [4], [5],
     [6], [7], [8], [9], [10]]

y = [10, 20, 30, 40, 50,
     60, 70, 80, 90, 100]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Actual:", y_test)
print("Predicted:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# New prediction
print("Prediction for 12:", model.predict([[12]])[0])

# Regression Line
line = model.predict(X)

plt.scatter(X, y)
plt.plot(X, line)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()




4)Demonstrate Data visualization.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = input("Enter CSV file name: ")
df = pd.read_csv(file)

print("\nData Loaded Successfully!")
print(df.head())

print("\n--- DATA VISUALIZATION MENU ---")
print("1. Histogram")
print("2. Bar Chart")
print("3. Line Chart")
print("4. Scatter Plot")
print("5. Box Plot")
print("6. Pie Chart")
print("7. Heatmap")
print("8. Exit")

while True:
    choice = int(input("\nEnter your choice: "))

    if choice == 1:
        sns.histplot(df["Marks"], kde=True)
        plt.title("Marks Distribution")
        plt.show()

    elif choice == 2:
        sns.barplot(x="Name", y="Marks", data=df)
        plt.title("Marks by Student")
        plt.xticks(rotation=45)
        plt.show()

    elif choice == 3:
        plt.plot(df["Name"], df["Marks"], marker="o")
        plt.title("Marks Trend")
        plt.xticks(rotation=45)
        plt.show()

    elif choice == 4:
        sns.scatterplot(x="StudyHours", y="Marks", data=df)
        plt.title("Study Hours vs Marks")
        plt.show()

    elif choice == 5:
        sns.boxplot(y=df["Marks"])
        plt.title("Marks Box Plot")
        plt.show()

    elif choice == 6:
        df["Gender"].value_counts().plot.pie(autopct="%1.1f%%")
        plt.title("Gender Distribution")
        plt.ylabel("")
        plt.show()

    elif choice == 7:
        sns.heatmap(df.select_dtypes("number").corr(), annot=True)
        plt.title("Correlation Heatmap")
        plt.show()

    elif choice == 8:
        print("Program Ended")
        break

    else:
        print("Invalid Choice")





OUTPUT
Enter CSV file name:  s.csv

Data Loaded Successfully!
    Name  Age  Marks  Attendance  StudyHours  Gender
0   John   20     85          90           5    Male
1   Sara   21     92          95           7  Female
2   Mike   20     70          80           3    Male
3   Anna   22     88          92           6  Female
4  David   21     65          75           2    Male







5)Implementation of classification models-Naïve bayes clasiifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Attendance (%) and Marks (%)
X = [
    [90, 85],
    [85, 80],
    [95, 90],
    [60, 55],
    [65, 60],
    [70, 65],
    [88, 82],
    [55, 50]
]

y = [
    'Pass', 'Pass', 'Pass', 'Fail',
    'Fail', 'Fail', 'Pass', 'Fail'
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# New student prediction
print("New Student:", model.predict([[85, 78]])[0])








6) Implementation of classification models-Decision trees
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Attendance (%) and Marks (%)
X = [
    [90, 85], [85, 80], [95, 90],
    [60, 55], [65, 60], [70, 65],
    [88, 82], [55, 50]
]

y = [
    'Pass', 'Pass', 'Pass',
    'Fail', 'Fail', 'Fail',
    'Pass', 'Fail'
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# New student prediction
print("New Student:", model.predict([[85, 78]])[0])





7 ) Implementation of classification models-Random forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Attendance (%) and Marks (%)
X = [
    [90, 85], [85, 80], [95, 90],
    [60, 55], [65, 60], [70, 65],
    [88, 82], [55, 50]
]

y = [
    'Pass', 'Pass', 'Pass',
    'Fail', 'Fail', 'Fail',
    'Pass', 'Fail'
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train model
model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# New student prediction
print("New Student:", model.predict([[85, 78]])[0])




