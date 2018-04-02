import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle as cPickle
from sklearn import metrics
from pprint import pprint
import json
import sys

#path = '/afs/cs.pitt.edu/projects/kovashka/mingda2/methods/svm'
path = "./"
feature_size = 4096
target = sys.argv[1]

#with open(os.path.join(path, 'average_fc7_features.pkl'), 'rb') as fp:
with open("./average_fc7_features_new.pkl", "rb") as fp:
    data = cPickle.load(fp)

with open(os.path.join(path, 'train_2087.txt'), 'r') as fp:
    train_id = fp.read().rstrip().split()

with open(os.path.join(path, 'valid_695.txt'), 'r') as fp:
    valid_id = fp.read().rstrip().split()

with open(os.path.join(path, 'test_695.txt'), 'r') as fp:
    test_id = fp.read().rstrip().split()

train_id = train_id + valid_id


#######################################################
## Load funny exciting score
#######################################################

def loadLabel(label, threshold=0.7):
  with open('final_%s_score.csv' % label, 'r') as fp:
    content = fp.read().strip().split('\n')
  mapping = {}
  for line in content:
    vid, label = line.split(', ')
    if float(label) > threshold:
      mapping[vid] = 1
    elif float(label) < 1 - threshold:
      mapping[vid] = 0
  return mapping

def loadJsonLabel(label):
  with open('video_%s_clean.json' % label, 'r') as fp:
    content = json.load(fp)
  return content

labels = loadJsonLabel(target)

train_X = np.empty([len(train_id), feature_size])
train_y = np.empty(len(train_id))
test_X = np.empty([len(test_id), feature_size])
test_y = np.empty(len(test_id))
train_X.fill(np.nan)
train_y.fill(np.nan)
test_X.fill(np.nan)
test_y.fill(np.nan)

i = 0
for vid in train_id:
  if vid in labels:
    train_X[i, :] = data[vid]
    train_y[i] = labels[vid] - 1
    i += 1

train_X = train_X[:i, :]
train_y = train_y[:i]

j = 0
for vid in test_id:
  if vid in labels:
    test_X[j, :] = data[vid]
    test_y[j] = labels[vid] - 1
    j += 1

test_X = test_X[:j, :]
test_y = test_y[:j]

assert(not (train_X == np.nan).any())
assert(not (train_y == np.nan).any())
assert(not (test_X == np.nan).any())
assert(not (test_y == np.nan).any())

print('data loaded.')

#######################################################
## Select best hyper-parameter C from train
#######################################################

# Cs = [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1, 2, 5, 8, 10,
#       12, 15, 20, 25, 50, 100]
# Cs = np.arange(1, 30, 1)
# results = {}

# for C in Cs:
C = 15
clf = svm.SVC(decision_function_shape='ovr', C=C)

print('C: %f' % C)
clf.fit(train_X, train_y)
predict_y = clf.predict(test_X)

# target_names = ['not %s' % target, '%s' % target]

print(classification_report(test_y, predict_y, digits=3))
print('Accuracy: %f' % accuracy_score(test_y, predict_y))
# print('Dummy Guess: %f' % (1.0 * sum(train_y) / len(train_y)))


# print('C: %f' % C)
# result = {}
# scores = cross_val_score(clf, train_X, train_y,
#                          cv=10, scoring='f1_weighted', n_jobs=-1)
# print('Mean F1 CV Score: %f' % scores.mean())
# result['F1_Mean'] = scores.mean()
# result['F1_Detail'] = scores.tolist()
# result['F1_Std'] = scores.std()
#
# scores = cross_val_score(clf, train_X, train_y,
#                          cv=10, scoring='accuracy', n_jobs=-1)
# print('Mean Accuracy CV Score: %f' % scores.mean())
# result['Accuracy_Mean'] = scores.mean()
# result['Accuracy_Detail'] = scores.tolist()
# result['Accuracy_Std'] = scores.std()

#   results[C] = result


# with open('output/%s_C_15.json' % target, 'w') as fp:
#   json.dump(result, fp)
