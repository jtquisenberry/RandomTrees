from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from functools import reduce
import random
import numpy
from sklearn import tree
from matplotlib import pyplot as plt


def generate_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=5, min_samples_leaf=3)
    rf.fit(X_train, y_train)
    print("rf score ", rf.score(X_test, y_test))
    return rf

def add_tree(forest: RandomForestClassifier, tree):
    forest.estimators_.append(tree)
    forest.n_estimators += 1
    return forest

iris = load_iris()
X, y = iris.data[:, [0,1,2]], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

trees = list()
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

single_tree: DecisionTreeClassifier = rf.estimators_[8]
print(single_tree.tree_.n_node_samples)
print(single_tree.tree_.weighted_n_node_samples)
tree.plot_tree(single_tree)
plt.show()
a = 1



seeds = [x for x in range(10)]
for seed in seeds:
    random.seed(seed)
    t = DecisionTreeClassifier()
    t.random_state = seed
    t.fit(X_train, y_train)
    labels = t.predict(X_test)
    print(labels[:37])
    trees.append(t)
    add_tree(rf, t)

#del rf.estimators_[0]
#rf.n_estimators -= 1
labels = rf.predict(X_test)
print(labels[:37])
a = 1






# in the line below, we create 10 random forest classifier models
rfs = [generate_rf(X_train, y_train, X_test, y_test) for i in range(10)]
# in this step below, we combine the list of random forest models into one giant model
rf_combined = reduce(combine_rfs, rfs)
# the combined model scores better than *most* of the component models
print("rf combined score", rf_combined.score(X_test, y_test))
a = 1

