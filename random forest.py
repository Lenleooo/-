import pandas as pd
url = './wine21'
df = pd.read_csv(url, header = None)
df.columns = ['s', 'd', 'e', 'f',
              'g', 'h', 'i',
              'j', 'k', 'l',
              'm', 'n', 'o', 'p','q','r']

import numpy as np

print(np.unique(df['s']))
print(df.info())

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train.astype('int'))

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


