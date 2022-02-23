
import pandas as pd
import pickle

#reading dataset
penguins = pd.read_csv('penguins_cleaned.csv')

#creating copy
df = penguins.copy()
target = 'species'
encode = ['sex', 'island']


for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

#separate X and y

X = df.drop('species', axis=1)
Y= df['species']

#build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,Y)

#saving model
pickle.dump(clf,open('penguin_clf.pkl', 'wb'))

