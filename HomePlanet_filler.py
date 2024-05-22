def HomePlanetFiller():
    import random
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from sklearn.model_selection import train_test_split


    #for test delete later
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    from catboost import CatBoostClassifier
    #for test delete later

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    import warnings
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from random_word import RandomWords


    warnings.filterwarnings("ignore")

    # read data
    df = pd.read_csv("train.csv")
    df = df.dropna()

    #df.reset_index(drop=True, inplace=True)

    # divide cabin data
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split('/', expand=True)
    #print(df["Deck"].value_counts(normalize=True))

    ##################################################################fillna##############################################################
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")
    ################################################################### EXPERIMENTS ###########################################################
    ############################################################REVERT BACK IF NECCECARY#######################################################
    df["pass_group_seat"] = df["PassengerId"].str.split('_').str[-1]
    df["pass_group_seat"] = pd.to_numeric(df["pass_group_seat"], errors='coerce').fillna(df["PassengerId"])
    df["total_spending"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    df["vr_spa_room"] = df["VRDeck"] + df["Spa"] + df["RoomService"]
    df["food_shopping"] = df["FoodCourt"] + df["ShoppingMall"]

    def suppress_outliars(df,col):
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1

        low_level=Q1-1.5*IQR
        high_level=Q3+1.5*IQR

        df.loc[(df[col]<=low_level),col]=(low_level)
        df.loc[(df[col]>high_level),col]=(high_level)

        return df

    # drop irrelevant columns
    df.drop(columns=["PassengerId","Cabin","Name"],
            inplace=True)

    cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
    num_cols=[col for col in df.columns if df[col].dtypes in ["float64","int64"]]
    num_but_cat = [col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ["float64","int64"]]

    additional_cols_to_subtract = ['pass_group', 'Num', 'total_spending', 'food_spa']
    cols_to_subtract = num_but_cat + additional_cols_to_subtract

    num_cols = [col for col in num_cols if col not in cols_to_subtract]
    cat_cols = [col for col in (cat_col + num_but_cat) if col != "HomePlanet"]

    numeric_cols = num_cols
    for col in numeric_cols:
        suppress_outliars(df,col)

    ################################################################### OUTLIARS ##############################################################

    X = df.drop("HomePlanet", axis=1)
    y = df["HomePlanet"]
    X.reset_index(drop=True, inplace=True)

    ################################################################### DROP COLS ##############################################################
    ################################################################### ENCODER ##############################################################
    # one hot encoder
    columns_to_encode = cat_cols
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    encoded_data = encoder.fit_transform(X[columns_to_encode])

    encoded_data = pd.DataFrame(encoded_data,
                              columns=encoder.get_feature_names_out(columns_to_encode))
    X=(pd.concat([X.drop(columns_to_encode,axis=1),encoded_data],axis=1))
    ################################################################### ENCODER ##############################################################
    ################################################################### ENCODER ##############################################################
    ################################################################### ENCODER ##############################################################
    print("HomePlanet Train Data working.....")

    ################################################################### LEARNING ##############################################################

    """
    # learning
    accuracy_dict = {}
    for i in range(19, 21):
        i = i / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=i, random_state=42
        )
    
        model = CatBoostClassifier(verbose=False)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
    
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {i}= {accuracy}")
    
        accuracy_dict[i] = accuracy
    
    sorted_accuracy = dict(sorted(accuracy_dict.items(), key=lambda item: item[1]))
    print(sorted_accuracy)
    
    first_key = list(sorted_accuracy)[-1]
    print(first_key)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.19, random_state=42)


    model = CatBoostClassifier(verbose=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    ################################################################### LEARNING ##############################################################


    original_df = pd.read_csv("train.csv")

    missing_homeplanet_df = original_df[original_df['HomePlanet'].isna()]

    # Preprocess missing_homeplanet_df: Divide cabin data, create new features, etc.
    missing_homeplanet_df[["Deck", "Num", "Side"]] = missing_homeplanet_df["Cabin"].str.split('/', expand=True)
    missing_homeplanet_df["pass_group_seat"] = missing_homeplanet_df["PassengerId"].str.split('_').str[-1]
    missing_homeplanet_df["pass_group_seat"] = pd.to_numeric(missing_homeplanet_df["pass_group_seat"], errors='coerce').fillna(missing_homeplanet_df["PassengerId"])
    missing_homeplanet_df["total_spending"] = missing_homeplanet_df["RoomService"] + missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["VRDeck"]
    missing_homeplanet_df["vr_spa_room"] = missing_homeplanet_df["VRDeck"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["RoomService"]
    missing_homeplanet_df["food_shopping"] = missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"]
    missing_homeplanet_df.drop(columns=["PassengerId", "Cabin", "Name"], inplace=True)

    age_counts_normalized = missing_homeplanet_df["Age"].value_counts(normalize=True)
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
    missing_homeplanet_df["Age"].fillna(selected_age, inplace=True)

    numeric_cols_fillna = ["RoomService", "FoodCourt",
                    "ShoppingMall", "Spa", "VRDeck"]

    for col in numeric_cols_fillna:
        missing_homeplanet_df[col].fillna(missing_homeplanet_df[col].mean(), inplace=True)

    r = RandomWords()
    if 'Name' in missing_homeplanet_df.columns:
        r = RandomWords()
        missing_homeplanet_df['Name'] = missing_homeplanet_df['Name'].fillna(lambda x: f'x {r.get_random_word()}')
    def per_calculator(df, col, value):
        return (df[col] == value).sum() / len(df) * 100

    # per for train
    percentage_true = per_calculator(missing_homeplanet_df, "Transported", True)
    percentage_vip = per_calculator(df, "VIP", True)
    #percentage_homeplanet_earth = per_calculator(missing_homeplanet_df, "HomePlanet", "Earth")
    #percentage_homeplanet_europa = per_calculator(missing_homeplanet_df, "HomePlanet", "Europa")
    percentage_CryoSleep = per_calculator(missing_homeplanet_df, "CryoSleep", True)
    percentage_Destination_TRAPPIST = per_calculator(missing_homeplanet_df, "Destination", "TRAPPIST-1e")
    percentage_Destination_PSO = per_calculator(missing_homeplanet_df, "Destination", "PSO J318.5-22")
    percentage_Destination_55_Cancri_e = per_calculator(missing_homeplanet_df, "Destination", "55 Cancri e")


    def random_state_selector(transported):
        random_num = random.randint(0, 1000)
        if random_num<=transported*10:
            state="True"
        else:
            state="False"
        return state
    def random_CryoSleep_selector(cryo):
        random_num = random.randint(0, 1000)
        if random_num <= cryo * 10:
            CryoSleep_state = True
        else:
            CryoSleep_state = False
        return CryoSleep_state

    def random_Destination_selector(trappist,pso):
        random_num = random.randint(0, 1000)
        if random_num<=trappist*10:
            destination="TRAPPIST-1e"
        elif random_num>trappist*10 and (random_num<=trappist*10+pso*10):
            destination="PSO J318.5-22"
        else:
            destination="55 Cancri e"
        return destination

    def random_side_selector(side_selector):
        random_num = random.randint(0, 1000)
        if random_num<=side_selector*10:
            side="P"
        else:
            side="S"
        return side
    def random_deck_selector():
        random_num = random.randint(0, 10000)
        if random_num<=3289:
            deck="F"
        elif random_num>3289 and random_num<=(3289+3012):
            deck="G"
        elif random_num>(3289+3012) and random_num<=(3289+3012+1031):
            deck = "E"
        elif random_num>(3289+3012+1031) and random_num<=(3289+3012+1031+917):
            deck = "B"
        elif random_num>(3289+3012+1031+917) and random_num<=(3289+3012+1031+917+879):
            deck = "C"
        elif random_num>(3289+3012+1031+917+879) and random_num<=(3289+3012+1031+917+562):
            deck = "D"
        elif random_num>(3289+3012+1031+917+879+562) and random_num<=(3289+3012+1031+917+562+301):
            deck = "A"
        else:
            deck = "T"
        return deck

    missing_homeplanet_df["Transported"] = missing_homeplanet_df["Transported"].fillna(random_state_selector(percentage_true))
    missing_homeplanet_df["CryoSleep"] = missing_homeplanet_df["CryoSleep"].fillna(random_CryoSleep_selector(percentage_CryoSleep))
    missing_homeplanet_df["Destination"] = missing_homeplanet_df["Destination"].fillna(random_Destination_selector(percentage_Destination_TRAPPIST,percentage_Destination_PSO))
    most_common_Deck = df["Deck"].mode()[0]
    percentage_side = (df["Side"] == "P").sum() / len(df) * 100
    most_common_Deck_test = missing_homeplanet_df["Deck"].mode()[0]
    missing_homeplanet_df["Deck"] = missing_homeplanet_df["Deck"].fillna(random_deck_selector())
    missing_homeplanet_df["Side"] = missing_homeplanet_df["Side"].fillna(random_side_selector(percentage_side))
    missing_homeplanet_df["VIP"] = missing_homeplanet_df["VIP"].fillna(random_state_selector(percentage_vip))

    most_common_Num = missing_homeplanet_df["Num"].mode()[0]
    missing_homeplanet_df["Num"] = missing_homeplanet_df["Num"].fillna(most_common_Num)

    missing_homeplanet_df["total_spending"] = missing_homeplanet_df["RoomService"] + missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["VRDeck"]
    missing_homeplanet_df["vr_spa_room"] = missing_homeplanet_df["VRDeck"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["RoomService"]
    missing_homeplanet_df["food_shopping"] = missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"]


    missing_homeplanet_df.drop(["HomePlanet"], axis=1, inplace=True)


    missing_homeplanet_df.reset_index(drop=True, inplace=True)
    missing_homeplanet_df[columns_to_encode] = missing_homeplanet_df[columns_to_encode].astype(str)
    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data_test = encoder.fit_transform(missing_homeplanet_df[columns_to_encode])
    encoded_data_test = pd.DataFrame(encoded_data_test, columns=encoder.get_feature_names_out(columns_to_encode))
    missing_data_df = pd.concat([missing_homeplanet_df.drop(columns_to_encode, axis=1), encoded_data_test], axis=1)


    # Predict missing 'HomePlanet' values
    missing_homeplanet_df.reset_index(drop=True, inplace=True)
    missing_homeplanet_df[columns_to_encode] = missing_homeplanet_df[columns_to_encode].astype(str)
    encoded_data_test = encoder.transform(missing_homeplanet_df[columns_to_encode])
    encoded_data_test = pd.DataFrame(encoded_data_test, columns=encoder.get_feature_names_out(columns_to_encode))
    missing_data_df = pd.concat([missing_homeplanet_df.drop(columns_to_encode, axis=1), encoded_data_test], axis=1)

    # Predict missing 'HomePlanet' values
    predictions = model.predict(missing_data_df)

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame(predictions, columns=["HomePlanet"])

    # Update the 'HomePlanet' column in missing_homeplanet_df with the predictions
    missing_homeplanet_df["HomePlanet"] = predictions_df["HomePlanet"]

    # Check if missing_homeplanet_df is correctly filled

    # Concatenate the DataFrame such that rows with missing 'HomePlanet' values are moved to the top
    original_df_sorted = pd.concat([
        original_df[original_df['HomePlanet'].isnull()],  # Rows with missing 'HomePlanet' values
        original_df.dropna(subset=['HomePlanet'])        # Rows with non-missing 'HomePlanet' values
    ])

    # Reset index
    original_df_sorted.reset_index(drop=True, inplace=True)

    # Update the 'HomePlanet' column with predictions for missing values
    original_df_sorted.loc[original_df_sorted['HomePlanet'].isnull(), 'HomePlanet'] = missing_homeplanet_df['HomePlanet']

    # Save the updated DataFrame to a new CSV file
    original_df_sorted.to_csv("train_filled_home.csv", index=False)
    print(f"HomePlanet Train Data DONE with %{accuracy*100} accuracy")












    import random
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    #for test delete later
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    from catboost import CatBoostClassifier
    #for test delete later

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from random_word import RandomWords


    # read data
    df = pd.read_csv("test.csv")
    df = df.dropna()

    #df.reset_index(drop=True, inplace=True)

    # divide cabin data
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split('/', expand=True)

    ##################################################################fillna##############################################################
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")
    ################################################################### EXPERIMENTS ###########################################################
    ############################################################REVERT BACK IF NECCECARY#######################################################
    df["pass_group_seat"] = df["PassengerId"].str.split('_').str[-1]
    df["pass_group_seat"] = pd.to_numeric(df["pass_group_seat"], errors='coerce').fillna(df["PassengerId"])
    df["total_spending"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    df["vr_spa_room"] = df["VRDeck"] + df["Spa"] + df["RoomService"]
    df["food_shopping"] = df["FoodCourt"] + df["ShoppingMall"]

    def suppress_outliars(df,col):
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1

        low_level=Q1-1.5*IQR
        high_level=Q3+1.5*IQR

        df.loc[(df[col]<=low_level),col]=(low_level)
        df.loc[(df[col]>high_level),col]=(high_level)

        return df

    # drop irrelevant columns
    df.drop(columns=["PassengerId","Cabin","Name"],
            inplace=True)

    cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
    num_cols=[col for col in df.columns if df[col].dtypes in ["float64","int64"]]

    num_but_cat = [col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ["float64","int64"]]

    additional_cols_to_subtract = ['pass_group', 'Num', 'total_spending', 'food_spa']
    cols_to_subtract = num_but_cat + additional_cols_to_subtract

    num_cols = [col for col in num_cols if col not in cols_to_subtract]
    cat_cols = [col for col in (cat_col + num_but_cat) if col != "HomePlanet"]

    numeric_cols = num_cols
    for col in numeric_cols:
        suppress_outliars(df,col)

    ################################################################### OUTLIARS ##############################################################

    X = df.drop("HomePlanet", axis=1)
    y = df["HomePlanet"]
    X.reset_index(drop=True, inplace=True)

    ################################################################### DROP COLS ##############################################################
    ################################################################### ENCODER ##############################################################
    # one hot encoder
    columns_to_encode = cat_cols

    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    encoded_data = encoder.fit_transform(X[columns_to_encode])

    encoded_data = pd.DataFrame(encoded_data,
                              columns=encoder.get_feature_names_out(columns_to_encode))

    X=(pd.concat([X.drop(columns_to_encode,axis=1),encoded_data],axis=1))
    ################################################################### ENCODER ##############################################################
    X.drop(columns=["Side_P"], inplace=True)
    X.drop(columns=["pass_group_seat_6"], inplace=True)
    X.drop(columns=["pass_group_seat_7"], inplace=True)
    X.drop(columns=["pass_group_seat_8"], inplace=True)
    ################################################################### ENCODER ##############################################################
    ################################################################### ENCODER ##############################################################


    ################################################################### LEARNING ##############################################################

    """
    # learning
    accuracy_dict = {}
    for i in range(15, 31):
        i = i / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=i, random_state=42
        )
    
        model = CatBoostClassifier(verbose=False)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
    
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {i}= {accuracy}")
    
        accuracy_dict[i] = accuracy
    
    sorted_accuracy = dict(sorted(accuracy_dict.items(), key=lambda item: item[1]))
    print(sorted_accuracy)
    
    first_key = list(sorted_accuracy)[-1]
    print(first_key)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.28, random_state=42)


    model = CatBoostClassifier(verbose=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    ################################################################### LEARNING ##############################################################


    original_df = pd.read_csv("test.csv")

    missing_homeplanet_df = original_df[original_df['HomePlanet'].isna()]

    # Preprocess missing_homeplanet_df: Divide cabin data, create new features, etc.
    missing_homeplanet_df[["Deck", "Num", "Side"]] = missing_homeplanet_df["Cabin"].str.split('/', expand=True)
    missing_homeplanet_df["pass_group_seat"] = missing_homeplanet_df["PassengerId"].str.split('_').str[-1]
    missing_homeplanet_df["pass_group_seat"] = pd.to_numeric(missing_homeplanet_df["pass_group_seat"], errors='coerce').fillna(missing_homeplanet_df["PassengerId"])
    missing_homeplanet_df["total_spending"] = missing_homeplanet_df["RoomService"] + missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["VRDeck"]
    missing_homeplanet_df["vr_spa_room"] = missing_homeplanet_df["VRDeck"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["RoomService"]
    missing_homeplanet_df["food_shopping"] = missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"]
    missing_homeplanet_df.drop(columns=["PassengerId", "Cabin", "Name"], inplace=True)

    age_counts_normalized = missing_homeplanet_df["Age"].value_counts(normalize=True)
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
    missing_homeplanet_df["Age"].fillna(selected_age, inplace=True)

    numeric_cols_fillna = ["RoomService", "FoodCourt",
                    "ShoppingMall", "Spa", "VRDeck"]

    for col in numeric_cols_fillna:
        missing_homeplanet_df[col].fillna(missing_homeplanet_df[col].mean(), inplace=True)

    r = RandomWords()
    if 'Name' in missing_homeplanet_df.columns:
        r = RandomWords()
        missing_homeplanet_df['Name'] = missing_homeplanet_df['Name'].fillna(lambda x: f'x {r.get_random_word()}')
    def per_calculator(df, col, value):
        return (df[col] == value).sum() / len(df) * 100

    # per for train

    percentage_vip = per_calculator(df, "VIP", True)
    #percentage_homeplanet_earth = per_calculator(missing_homeplanet_df, "HomePlanet", "Earth")
    #percentage_homeplanet_europa = per_calculator(missing_homeplanet_df, "HomePlanet", "Europa")
    percentage_CryoSleep = per_calculator(missing_homeplanet_df, "CryoSleep", True)
    percentage_Destination_TRAPPIST = per_calculator(missing_homeplanet_df, "Destination", "TRAPPIST-1e")
    percentage_Destination_PSO = per_calculator(missing_homeplanet_df, "Destination", "PSO J318.5-22")
    percentage_Destination_55_Cancri_e = per_calculator(missing_homeplanet_df, "Destination", "55 Cancri e")


    def random_state_selector(transported):
        random_num = random.randint(0, 1000)
        if random_num<=transported*10:
            state="True"
        else:
            state="False"
        return state
    def random_CryoSleep_selector(cryo):
        random_num = random.randint(0, 1000)
        if random_num <= cryo * 10:
            CryoSleep_state = True
        else:
            CryoSleep_state = False
        return CryoSleep_state

    def random_Destination_selector(trappist,pso):
        random_num = random.randint(0, 1000)
        if random_num<=trappist*10:
            destination="TRAPPIST-1e"
        elif random_num>trappist*10 and (random_num<=trappist*10+pso*10):
            destination="PSO J318.5-22"
        else:
            destination="55 Cancri e"
        return destination

    def random_side_selector(side_selector):
        random_num = random.randint(0, 1000)
        if random_num<=side_selector*10:
            side="P"
        else:
            side="S"
        return side
    def random_deck_selector():
        random_num = random.randint(0, 10000)
        if random_num<=3289:
            deck="F"
        elif random_num>3289 and random_num<=(3289+3012):
            deck="G"
        elif random_num>(3289+3012) and random_num<=(3289+3012+1031):
            deck = "E"
        elif random_num>(3289+3012+1031) and random_num<=(3289+3012+1031+917):
            deck = "B"
        elif random_num>(3289+3012+1031+917) and random_num<=(3289+3012+1031+917+879):
            deck = "C"
        elif random_num>(3289+3012+1031+917+879) and random_num<=(3289+3012+1031+917+562):
            deck = "D"
        elif random_num>(3289+3012+1031+917+879+562) and random_num<=(3289+3012+1031+917+562+301):
            deck = "A"
        else:
            deck = "T"
        return deck

    missing_homeplanet_df["CryoSleep"] = missing_homeplanet_df["CryoSleep"].fillna(random_CryoSleep_selector(percentage_CryoSleep))
    missing_homeplanet_df["Destination"] = missing_homeplanet_df["Destination"].fillna(random_Destination_selector(percentage_Destination_TRAPPIST,percentage_Destination_PSO))
    most_common_Deck = df["Deck"].mode()[0]
    percentage_side = (df["Side"] == "P").sum() / len(df) * 100
    most_common_Deck_test = missing_homeplanet_df["Deck"].mode()[0]
    missing_homeplanet_df["Deck"] = missing_homeplanet_df["Deck"].fillna(random_deck_selector())
    missing_homeplanet_df["Side"] = missing_homeplanet_df["Side"].fillna(random_side_selector(percentage_side))
    missing_homeplanet_df["VIP"] = missing_homeplanet_df["VIP"].fillna(random_state_selector(percentage_vip))

    most_common_Num = missing_homeplanet_df["Num"].mode()[0]
    missing_homeplanet_df["Num"] = missing_homeplanet_df["Num"].fillna(most_common_Num)

    missing_homeplanet_df["total_spending"] = missing_homeplanet_df["RoomService"] + missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["VRDeck"]
    missing_homeplanet_df["vr_spa_room"] = missing_homeplanet_df["VRDeck"] + missing_homeplanet_df["Spa"] + missing_homeplanet_df["RoomService"]
    missing_homeplanet_df["food_shopping"] = missing_homeplanet_df["FoodCourt"] + missing_homeplanet_df["ShoppingMall"]


    missing_homeplanet_df.drop(["HomePlanet"], axis=1, inplace=True)


    missing_homeplanet_df.reset_index(drop=True, inplace=True)
    missing_homeplanet_df[columns_to_encode] = missing_homeplanet_df[columns_to_encode].astype(str)
    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data_test = encoder.fit_transform(missing_homeplanet_df[columns_to_encode])
    encoded_data_test = pd.DataFrame(encoded_data_test, columns=encoder.get_feature_names_out(columns_to_encode))
    missing_data_df = pd.concat([missing_homeplanet_df.drop(columns_to_encode, axis=1), encoded_data_test], axis=1)


    # Predict missing 'HomePlanet' values
    missing_homeplanet_df.reset_index(drop=True, inplace=True)
    missing_homeplanet_df[columns_to_encode] = missing_homeplanet_df[columns_to_encode].astype(str)
    encoded_data_test = encoder.transform(missing_homeplanet_df[columns_to_encode])
    encoded_data_test = pd.DataFrame(encoded_data_test, columns=encoder.get_feature_names_out(columns_to_encode))
    missing_data_df = pd.concat([missing_homeplanet_df.drop(columns_to_encode, axis=1), encoded_data_test], axis=1)

    # Predict missing 'HomePlanet' values
    predictions = model.predict(missing_data_df)

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame(predictions, columns=["HomePlanet"])

    # Update the 'HomePlanet' column in missing_homeplanet_df with the predictions
    missing_homeplanet_df["HomePlanet"] = predictions_df["HomePlanet"]


    # Concatenate the DataFrame such that rows with missing 'HomePlanet' values are moved to the top
    original_df_sorted = pd.concat([
        original_df[original_df['HomePlanet'].isnull()],  # Rows with missing 'HomePlanet' values
        original_df.dropna(subset=['HomePlanet'])        # Rows with non-missing 'HomePlanet' values
    ])

    # Reset index
    original_df_sorted.reset_index(drop=True, inplace=True)

    # Update the 'HomePlanet' column with predictions for missing values
    original_df_sorted.loc[original_df_sorted['HomePlanet'].isnull(), 'HomePlanet'] = missing_homeplanet_df['HomePlanet']

    # Save the updated DataFrame to a new CSV file
    original_df_sorted.to_csv("test_filled_home.csv", index=False)
    print(f"HomePlanet Test Filling DONE with %{accuracy*100} accuracy")



