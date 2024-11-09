import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class EvalSequenceAnalyser():
    def __init__(self):
        # X and Y form our training data. We fit a model to predict Y given X
        X = []
        Y = []

        # The four games used to gather training data
        for game in ("sf_torch", "sf_weiss", "me_ege", "me_random"):
            db = np.genfromtxt(f"training_data\\{game}.csv", delimiter=",")
            X_temp, Y_temp = db[:,1:], db[:,0]
            X.extend(X_temp)
            Y.extend(Y_temp)

        # Implement SVC model
        XTr, XTe, YTr, YTe = train_test_split(X, Y, random_state=2)

        # clf = GridSearchCV(SVC(), {"kernel": ["linear"],
        #                            "C":      [0.5, 1.0, 2.0, 5.0],
        #                            "decision_function_shape": ["ovo", "ovr"],
        #                            "probability": [True],
        #                            }, verbose=True).fit(XTr, YTr)
        # print(f"Accuracy linear:       {clf.score(XTe, YTe)}")

        # svm_predictions = clf.predict(XTe) 
        # svm_predictions_proba = clf.predict_proba(XTe) 

        # accuracy = clf.score(XTe, YTe)

        # cm = confusion_matrix(YTe, svm_predictions)

        self.model = SVC(kernel = 'linear',
                         decision_function_shape='ovo',
                         probability=True).fit(XTr, YTr)
    
    # Takes an evaluation levels vector and classifies the trend, returning the classification
    # and the confidence in the prediction
    def classify(self, eval_sequence):
        prediction = self.model.predict([eval_sequence])[0]
        prediction_proba = self.model.predict_proba([eval_sequence])[0]
        return prediction, prediction_proba