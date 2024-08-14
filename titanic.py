import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from warnings import simplefilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ignore warnings
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
sns.set_style('darkgrid')
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["figure.facecolor"] = "#00000000"
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 50)

titanic = pd.read_csv("train.csv")
# titanic.dropna(subset=["Embarked"],inplace=True)

# Split data
train_data, val_data = train_test_split(titanic, test_size=0.25,random_state=42)
test_data = pd.read_csv("test.csv")
dumb_model = pd.read_csv("gender_submission.csv")["Survived"]

# get input and target columns
input_cols = ["Pclass"] + ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', "Ticket"]
target_col ="Survived"

train_inputs = train_data[input_cols].copy()
train_target = train_data[target_col].copy()

val_inputs = val_data[input_cols].copy()
val_target = val_data[target_col].copy()

test_inputs = test_data[input_cols].copy()


numeric_cols = list(train_inputs.select_dtypes(include=np.number).columns)
categ_cols = list(train_inputs.select_dtypes(exclude=np.number).columns)

# fill in gaps in categorical and numeric columns
imputer_numeric = SimpleImputer(strategy="mean")
imputer_categ = SimpleImputer(strategy="most_frequent")

imputer_numeric.fit(train_inputs[numeric_cols])
imputer_categ.fit(train_inputs[categ_cols])

train_inputs[categ_cols] = imputer_categ.transform(train_inputs[categ_cols])
val_inputs[categ_cols] = imputer_categ.transform(val_inputs[categ_cols])
test_inputs[categ_cols] = imputer_categ.transform(test_inputs[categ_cols])

train_inputs[numeric_cols] = imputer_numeric.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols]  =  imputer_numeric.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols]  =  imputer_numeric.transform(test_inputs[numeric_cols])

# Encode categorical columns
encoder = OneHotEncoder(sparse_output = False, handle_unknown="ignore")
encoder.fit(train_inputs[categ_cols])
encoded_categ = list(encoder.get_feature_names_out())

train_inputs[encoded_categ] = encoder.transform(train_inputs[categ_cols])
val_inputs[encoded_categ] = encoder.transform(val_inputs[categ_cols])
test_inputs[encoded_categ] = encoder.transform(test_inputs[categ_cols])

# Scale
scaler = MinMaxScaler()
scaler.fit(train_inputs[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# model = LogisticRegression(solver="liblinear")
model = RandomForestClassifier(n_estimators=250, n_jobs=-1, max_depth=200, random_state=42)
model.fit(train_inputs[numeric_cols + encoded_categ], train_target)


# print(pd.DataFrame(
#     {
#     "feature": numeric_cols + encoded_categ,
#     "weights" : list(model.coef_)[0]
#     }
# ))

train_pred = model.predict(train_inputs[numeric_cols + encoded_categ])
print(f"Accuracy on training data: {accuracy_score(train_target, train_pred)}")

val_pred = model.predict(val_inputs[numeric_cols + encoded_categ])

print(f"Accuracy on validation data: {accuracy_score(val_target, val_pred)}")
cnf_matrix = list(confusion_matrix(val_target, val_pred, normalize="true" ))
print(f"True Negatives : {cnf_matrix[0][0]} , False positives : {cnf_matrix[0][1]}")
print(f"False Negatives : {cnf_matrix[1][0]} , True positives : {cnf_matrix[1][1]}")



# percent of women who survived
women = val_data[val_data.Sex == 'female']["Survived"]

# Accuracy of dumb model
print(f"accuracy of dumb model in validation data: {sum(women) / len(women)}  ")



# My answer
test_pred = model.predict(test_inputs[numeric_cols + encoded_categ])
with open("answer.csv","w") as file:
    file.write(f"PassengerId,Survived\n")
    for ID, answer in zip(test_data["PassengerId"], test_pred):
        file.write(f"{ID},{answer}\n")



# print(f"accuracy : {accuracy_score(dumb_model,test_pred)}")


