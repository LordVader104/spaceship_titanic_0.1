import random
import pandas as pd
from random_word import RandomWords
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

#for test delete later
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer
#for test delete later

import warnings

from HomePlanet_filler import HomePlanetFiller
from CryoSleep_filler import CryoSleepFiller
from VIP_filler import VIPFiller
from Deck_filler import DeckFiller
from Destination_filler import DestinationFiller

warnings.filterwarnings("ignore")
# read data
"""
HomePlanetFiller()
CryoSleepFiller()
VIPFiller()
DeckFiller()
DestinationFiller()
"""
df = pd.read_csv("train_filled_Destination.csv")
test_data = pd.read_csv("test_filled_Destination.csv")
df["CryoSleep"].fillna("False")

print(df.columns)
print(df.describe().T)
print(df.info())

print("train empty count: ",df.isnull().sum())
print("test empty count: ",test_data.isnull().sum())
##################################################################fillna##############################################################
age_counts_normalized = df["Age"].value_counts(normalize=True)
age_counts_dict = age_counts_normalized.to_dict()

#chatgpt alg for selecting ages by per
def select_random_age(age_counts_dict):
    random_num = random.random()
    cumulative_percentage = 0
    for age, percentage in age_counts_dict.items():
        cumulative_percentage += percentage
        if random_num <= cumulative_percentage:
            if random.randint(0, 1):
                return age
            else:
                continue

selected_age = select_random_age(age_counts_dict)
df["Age"].fillna(selected_age, inplace=True)
test_data["Age"].fillna(selected_age, inplace=True)

numeric_cols_fillna = ["FoodCourt","ShoppingMall","RoomService","Spa", "VRDeck"]


def fill_numeric_cols(row, col):
    if row["CryoSleep"] == True:
        return 0
    elif row["HomePlanet"]=="Earth":
        if col=="RoomService" or col== "FoodCourt" or col=="ShoppingMall":
            return 700
        elif col=="Spa" or col=="VRDeck" :
            return 1000
    elif row["HomePlanet"]=="Mars":
        if col == "RoomService" or col=="ShoppingMall":
            return 5000
        elif col== "FoodCourt" or col== "VRDeck":
            return 0
        elif col=="Spa":
            return 700
    elif row["HomePlanet"]=="Europa":
        if col== "FoodCourt" or col=="Spa" or "VRDeck":
            return 5000
        elif col=="RoomService" or col=="ShoppingMall":
            return 0

for col in numeric_cols_fillna:
    df[col] = df.apply(lambda row: fill_numeric_cols(row, col) if pd.isnull(row[col]) else row[col], axis=1)

for col in numeric_cols_fillna:
    test_data[col] = test_data.apply(lambda row: fill_numeric_cols(row, col) if pd.isnull(row[col]) else row[col], axis=1)





r = RandomWords()
df['Name'] = df['Name'].fillna(lambda x: f'x {r.get_random_word()}')
test_data['Name'] = df['Name'].fillna(lambda x: f'x {r.get_random_word()}')

#df["Name"]=df["Name"].fillna("unknown")
def per_calculator(df, col, value):
    return (df[col] == value).sum() / len(df) * 100

# per for train
percentage_true = per_calculator(df, "Transported", True)

def random_state_selector(transported):
    random_num = random.randint(0, 1000)
    if random_num<=transported*10:
        state="True"
    else:
        state="False"
    return state

def random_side_selector(side_selector):
    random_num = random.randint(0, 1000)
    if random_num<=side_selector*10:
        side="P"
    else:
        side="S"
    return side

df["Transported"] = df["Transported"].fillna(random_state_selector(percentage_true))

# divide passengers into groups and drop irrelevant data
"""
df["pass_group"] = df["PassengerId"].str.split('_').str[0]
df["pass_group"] = pd.to_numeric(df["pass_group"], errors='coerce').fillna(test_data["PassengerId"])

test_data["pass_group"] = test_data["PassengerId"].str.split('_').str[0]
test_data["pass_group"] = pd.to_numeric(df["pass_group"], errors='coerce').fillna(test_data["PassengerId"])
"""
most_common_Num = df["Num"].mode()[0]
percentage_side = (df["Side"] == "P").sum() / len(df) * 100

most_common_Num_test = test_data["Num"].mode()[0]
percentage_side_test = (test_data["Side"] == "P").sum() / len(test_data) * 100

df["Num"] = df["Num"].fillna(most_common_Num)
df["Side"] = df["Side"].fillna(random_side_selector(percentage_side))

test_data["Num"] = test_data["Num"].fillna(most_common_Num_test)
test_data["Side"] = test_data["Side"].fillna(random_side_selector(percentage_side_test))

df["Num"] = pd.to_numeric(df["Num"], errors="coerce")
test_data["Num"] = pd.to_numeric(test_data["Num"], errors="coerce")



################################################################### EXPERIMENTS ###########################################################
############################################################REVERT BACK IF NECCECARY#######################################################

#seperate families
"""
df[['first_name', 'family']] = df['Name'].str.split(' ', n=1, expand=True)
df.drop(columns=["first_name"], inplace=True)

test_data[['first_name', 'family']] = test_data['Name'].str.split(' ', n=1, expand=True)
test_data.drop(columns=["first_name"], inplace=True)
"""
df["pass_group_seat"] = df["PassengerId"].str.split('_').str[-1]
df["pass_group_seat"] = pd.to_numeric(df["pass_group_seat"], errors='coerce').fillna(df["PassengerId"])

test_data["pass_group_seat"] = test_data["PassengerId"].str.split('_').str[-1]
test_data["pass_group_seat"] = pd.to_numeric(test_data["pass_group_seat"], errors='coerce').fillna(test_data["PassengerId"])

df["total_spending"]=df["RoomService"]+df["FoodCourt"]+df["ShoppingMall"]+df["Spa"]+df["VRDeck"]
df["vr_spa_room"]=df["VRDeck"]+df["Spa"]+df["RoomService"]
df["food_shopping"]=df["FoodCourt"]+df["ShoppingMall"]




test_data["total_spending"]=test_data["RoomService"]+test_data["FoodCourt"]+test_data["ShoppingMall"]+test_data["Spa"]+test_data["VRDeck"]
test_data["vr_spa_room"]=test_data["VRDeck"]+test_data["Spa"]+test_data["RoomService"]
test_data["food_shopping"]=test_data["FoodCourt"]+test_data["ShoppingMall"]

################################################################### EXPERIMENTS ###########################################################
############################################################REVERT BACK IF NECCECARY#######################################################

def suppress_outliars(df,col):
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1

    low_level=Q1-1.5*IQR
    high_level=Q3+1.5*IQR

    df.loc[(df[col]<=low_level),col]=(low_level)
    df.loc[(df[col]>high_level),col]=(high_level)

    return df
"""
####################################################visuals###################################################
pd.set_option("display.max_column",None)
pd.set_option("display.width",500)

row_count = df.shape[0]









spendings=["total_spending","RoomService","FoodCourt",
           "ShoppingMall","Spa","VRDeck","vr_spa_room","food_shopping"]
for col in spendings:
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=col, y="Transported", data=df)
    plt.xlabel(f"{col}")
    plt.ylabel("Transported")
    plt.title(f"Relationship between {col} and Transported Status")
    plt.yticks([0, 1], ['False', 'True'])  # Set y-tick labels
    plt.show()


bar_plot_list=["pass_group_seat","HomePlanet","Destination","VIP","Deck","Side"]
for col in bar_plot_list:

    sns.barplot(x=df[col], y=df['PassengerId'].count(),
                hue=df['Transported'], data=df, palette='husl',
                estimator=lambda x: len(x))
    plt.xlabel(f"{col}")
    plt.ylabel("Passenger Quantity")
    plt.title(f"Passenger Quantity by {col} and Transported Status")
    plt.legend(title='Transported', loc='upper right')
    plt.show()
"""

df.drop(columns=["PassengerId","Name"],
        inplace=True)
test_data.drop(columns=["PassengerId","Name"],
        inplace=True)

################################################################### OUTLIARS ##############################################################
cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
print(f"*************************CATEGORIC COLUMNS*************************")
print(f"{cat_col}")
num_cols=[col for col in df.columns if df[col].dtypes in ["float64","int64"]]
print(f"*************************NUMERIC COLUMNS*************************")
print(f"{num_cols}")
num_but_cat = [col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ["float64","int64"]]
print(f"*************************HIGH CARDINAL NUM COLUMNS*************************")
print(f"{num_but_cat}")

additional_cols_to_subtract = ['pass_group', 'Num', 'total_spending', 'food_spa']
cols_to_subtract = num_but_cat + additional_cols_to_subtract

num_cols = [col for col in num_cols if col not in cols_to_subtract]
print(f"*************************NUMERIC COLUMNS AFTER SUBSTRACTION*************************")
print(f"{num_cols}")
cat_cols = [col for col in (cat_col + num_but_cat) if col != "Transported"]
print(f"*************************REAL CAT COLUMNS*************************")
print(f"{cat_cols}")

print(num_cols)
numeric_cols = num_cols

"""for col in numeric_cols:
    suppress_outliars(df,col)

for col in numeric_cols:
    suppress_outliars(test_data,col)"""

"""  """

################################################################### OUTLIARS ##############################################################




X = df.drop("Transported", axis=1)
y = df["Transported"]

################################################################### ENCODER ##############################################################
# one hot encoder
columns_to_encode = cat_cols

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(X[columns_to_encode])

encoded_data = pd.DataFrame(encoded_data,
                          columns=encoder.get_feature_names_out(columns_to_encode))
X=(pd.concat([X.drop(columns_to_encode,axis=1),encoded_data],axis=1))
################################################################### ENCODER ##############################################################
X.drop(columns=["CryoSleep_nan"], inplace=True)
################################################################### ENCODER ##############################################################
# one hot encoder
encoder = OneHotEncoder(sparse_output=False)
encoded_data_test = encoder.fit_transform(test_data[columns_to_encode])

encoded_data_test = pd.DataFrame(encoded_data_test,
                            columns=encoder.get_feature_names_out(columns_to_encode))
test_data = pd.concat([test_data.drop(columns_to_encode, axis=1), encoded_data_test], axis=1)
################################################################### ENCODER ##############################################################
print("SpaceShip Titanic Crash Analysis working.....")

################################################################### LEARNING ##############################################################

# learning
accuracy_dict = {}

for i in range(15, 41):
    i = i / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=i, random_state=42
    )

    model = lgb.LGBMClassifier(verbose=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for %{i*100} test size= %{accuracy*100} accuracy")

    accuracy_dict[i] = accuracy

sorted_accuracy = dict(sorted(accuracy_dict.items(), key=lambda item: item[1]))
print(sorted_accuracy)

first_key = list(sorted_accuracy)[-1]
print(first_key)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=first_key, random_state=42)

model = lgb.LGBMClassifier(verbose=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy {accuracy}")
################################################################### LEARNING ##############################################################

#predicting
predictions = model.predict(test_data)
predictions_df = pd.DataFrame(predictions, columns=["Transported"])
test_data_with_predictions = pd.concat([test_data, predictions_df], axis=1)


test_data = pd.read_csv("test_filled_Destination.csv")
results_df = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions_df['Transported']
})
results_df.to_csv(f"test_with_predictions_with_%{accuracy*100}_accuracy.csv", index=False)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("its all done....")