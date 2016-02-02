__author__ = 'cedricdeboom'


import numpy as np
import sklearn.linear_model.logistic as logistic
import cPickle


def learn_lr(train_data, train_labels, save_to=None):
    print 'Start training logistic classifier'
    classifier = logistic.LogisticRegressionCV(cv=5, solver='sag', max_iter=300, n_jobs=10, verbose=1)
    classifier.fit(train_data, train_labels)

    if save_to is not None:
        print 'Saving...'
        f = open(save_to, 'wb')
        cPickle.dump(classifier, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'Done.'

    return classifier

def load_classifier(cfile):
    f = open(cfile, 'rb')
    c = cPickle.load(f)
    f.close()
    return c

def predict_lr(lr_classifier, test_data):
    return lr_classifier.predict(test_data)

def predict_proba_lr(lr_classifier, test_data):
    return lr_classifier.predict_proba(test_data)

def accuracy(predicted_labels, true_labels):
    return float(np.sum(predicted_labels == true_labels)) / float(len(true_labels))