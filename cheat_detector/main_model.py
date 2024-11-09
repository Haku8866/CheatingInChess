import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn import metrics as m
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import multiprocessing
import xgboost

class MainModel():
    def __init__(self, db=None, fast=False):
        self.fast = fast
        if db is None:
            db = np.genfromtxt(f"training_data/main_training.csv", delimiter=",")

        X = db[:,1:]
        Y = db[:,0]

        if fast:
            X = np.column_stack((X[:,:10], X[:,14:]))

        XTr, XTe, YTr, YTe = train_test_split(X, Y, test_size=0.2,random_state=0)

        #-------------- Testing various other models
        # fprs = []
        # tprs = []
        # threshs = []
        # names = ["Logistic Regression", "SVC (linear)","SVC (polynomial)","SVC (radial basis function)",
        #          "$K$-nearest neighbours", "Naive Bayes"]
        #
        # for x,model in enumerate([
        #     LogisticRegression(max_iter=10000, class_weight='balanced', solver='lbfgs'),
        #     SVC(kernel="linear", probability=True),
        #     SVC(kernel="poly", probability=True),
        #     SVC(kernel="rbf", probability=True),
        #     KNeighborsClassifier(),
        #     GaussianNB()
        #     ]):
        #     self.clf = model        
        #     self.clf = GridSearchCV(
        #         self.clf,
        #         {},
        #         verbose=1,
        #         n_jobs=multiprocessing.cpu_count() // 2,
        #     ).fit(XTr, YTr)
        #     y_pred = self.clf.predict(XTe)
        #     Y2 = self.clf.predict_proba(XTe)[:,1]
        #     print("Best: %f using %s" % (self.clf.best_score_, self.clf.best_params_))
        #     print("Accuracy:", m.accuracy_score(YTe, y_pred))
        #     fpr, tpr, thresh = m.roc_curve(YTe, Y2)
        #     plt.plot(fpr, tpr, label=names[x])
        #     fprs.append(fpr)
        #     tprs.append(tpr)
        #     threshs.append(threshs)
        #     print("ROC_AUC: ", roc_auc_score(YTe, self.clf.predict_proba(XTe)[:, 1]))
        #
        # plt.legend()
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.grid(True)
        # plt.title("RoC curves for various models")
        # plt.show()
        
        # XGBoost model with tuned hyper parameters
        xgb_model = xgboost.XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2,
                                          tree_method='hist',
                                          scale_pos_weight=len(Y[Y==0])/len(Y[Y==1]),
                                          max_depth=7,
                                          n_estimators=10,
                                          learning_rate=0.09,
                                          min_child_weight=5,
                                          gamma=0.4,
                                          reg_lambda=4
                                          ).fit(XTr, YTr)
        self.clf = xgb_model

        # --------------- Grid Search to find best hyper parameters
        # self.clf = GridSearchCV(
        #     xgb_model,
        #     {
        #         "max_depth": [i for i in range(1,11)],
        #         "n_estimators": [i*5 for i in range(1,11)],
        #         "gamma": [i*0.1 for i in range(1,11)],
        #         "lambda": [i for i in range(1, 11)]
        #     },
        #     verbose=1,
        #     n_jobs=multiprocessing.cpu_count() // 2,
        # ).fit(XTr, YTr)

        # ------------------- Evaluate performance
        # y_pred = self.clf.predict(XTe)
        # print("Accuracy:", m.accuracy_score(YTe, y_pred))
        # print("ROC_AUC: ", roc_auc_score(YTe, self.clf.predict_proba(XTe)[:, 1]))
    
    def classify(self, move):
        return self.clf.predict_proba([move.metrics(fast=self.fast)])[0,1]
