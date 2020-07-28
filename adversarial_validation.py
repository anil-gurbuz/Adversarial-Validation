import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


melb = pd.read_csv("listings_Melbourne.csv")
sydney = pd.read_csv("listings_Sydney.csv")


def pre_processer(df:pd.DataFrame, max_price:float, min_city_counts:int, columns:list):
    
    # Convert price into float
    df.price = df.price.str.slice(start=1, stop=-3).str.replace(",","").astype("float")
    
    # Remove outliers in price
    df = df.loc[df.price < 2000,]
    
    # Remove rare examples of city column
    city_counts = pd.DataFrame(df.city.value_counts())
    df = df.loc[df.city.isin(city_counts.loc[city_counts.city>200,].index.to_list()),]
    
    # Remove missing values in bedrooms and turn into integer
    df = df.loc[df.bedrooms.notna(),]
    df.bedrooms = df.bedrooms.astype("int").copy()
    
    # Select the required columns
    df = df[columns].reset_index(drop=True)
    
    return df



melb = pre_processer(melb, max_price=2000, min_city_counts=200, columns=["city","bedrooms","price"])
sydney = pre_processer(sydney, max_price=2000, min_city_counts=200, columns=["city","bedrooms","price"])

# Correct inconsistencies
melb.loc[melb.city == "Saint Kilda","city"] ="St Kilda"
sydney.loc[sydney.city == "Bondi","city"] ="Bondi Beach"




def adversarial_valid(train:pd.DataFrame, test: pd.DataFrame, columns:list, categorical_columns:list, plot_importance=True):
            
    # CREATE A TARGET SERIES WITH 0 VALUES CORRESPONDING TO TRAIN EXAMPLES WHEREAS VALUE 1 FOR TEST EXAMPLES
    target = np.hstack([np.zeros(train.shape[0]), np.ones(test.shape[0])])
    
    # Combine train and test set
    train = pd.concat([train,test],axis=0)
    train = train[columns]
    
    # Encode categorical columns as LGBM doesn't accept string type columns as input
    for col in categorical_columns:
        encoder=preprocessing.LabelEncoder()
        train[col] = encoder.fit_transform(train[col].to_list())
    
    
    # Create a new train and test set -- Not just melb train, sydney test.
    train, test, y_train, y_test = model_selection.train_test_split(train, target, test_size=0.33, random_state=1,
                                                                    shuffle=True)
    
    # Convert to LGBM Dataset to prepare LGBM training
    train = lgb.Dataset(train, label=y_train, categorical_feature=categorical_columns)
    test = lgb.Dataset(test, label=y_test, categorical_feature=categorical_columns)
    
    # Set the parameters. -- It is a 2-class classification task so objective is binary
    param = {'objective': 'binary',
             'learning_rate': 0.01,
             "boosting": "gbdt",
             "metric": 'auc',
             "verbosity": -1}
    
    # Train the classifier
    clf = lgb.train(param, train, num_boost_round=200, valid_sets=[train, test], verbose_eval=50, early_stopping_rounds=50)
    
    # Draw Feature importance graph
    
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(importance_type="gain"), clf.feature_name())), columns=['Feature Split Total Gain', 'Feature'])
    if not plot_importance:
        return clf, feature_imp
    else:
        plt.figure(figsize=(5, 5))
        sns.barplot(x="Feature Split Total Gain", y="Feature", data=feature_imp.sort_values(by="Feature Split Total Gain", ascending=False).head(100))
        plt.title('LightGBM - Feature Importance')
        plt.tight_layout()
        plt.show()

        return clf, feature_imp