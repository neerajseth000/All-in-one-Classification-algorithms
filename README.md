# All-in-one-Classification-algorithms
# This code will generate a table which consists of 10 different Classification Algorithm's result in terms of testing accuracy, Cohen Kappa Score and Matthew's correlation coefficient.

from sklearn.metrics import accuracy_score,matthews_corrcoef,confusion_matrix,cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from xgboost.sklearn import XGBRFClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split
xtrain2,xtest2,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=46)

# KNN Model
print("KNN")
knn2=KNeighborsClassifier()
pred_knn2=knn2.fit(xtrain2,ytrain).predict(xtest2)
knn_accuracy=accuracy_score(ytest,pred_knn2)
knn_kappa=cohen_kappa_score(ytest,pred_knn2)
knn_mcc=matthews_corrcoef(ytest,pred_knn2)

# Decision Tree
print("DecisionTree")
dtree2=DecisionTreeClassifier()
pred_dtree2 = dtree2.fit(xtrain2,ytrain).predict(xtest2)
dt_accuracy=accuracy_score(ytest,pred_dtree2)
dt_kappa=cohen_kappa_score(ytest,pred_dtree2)
dt_mcc=matthews_corrcoef(ytest,pred_dtree2)

#RF
print("RandomForest")
rf2 = RandomForestClassifier(criterion="entropy")
pred_rf2 = rf2.fit(xtrain2,ytrain).predict(xtest2)
rf_accuracy=accuracy_score(ytest,pred_rf2)
rf_kappa=cohen_kappa_score(ytest,pred_rf2)
rf_mcc=matthews_corrcoef(ytest,pred_rf2)

# Support Vector
print("SVC")
svc2 = SVC()
pred_svc2 =svc2.fit(xtrain2,ytrain).predict(xtest2)
svc_accuracy=accuracy_score(ytest,pred_svc2)
svc_kappa=cohen_kappa_score(ytest,pred_svc2)
svc_mcc=matthews_corrcoef(ytest,pred_svc2)

# NaiveBayes
print("Gaussian naive")
naiveClassifier2=GaussianNB()
pred_naiv2 = naiveClassifier2.fit(xtrain2,ytrain).predict(xtest2)
nb_accuracy=accuracy_score(ytest,pred_naiv2)
nb_kappa=cohen_kappa_score(ytest,pred_naiv2)
nb_mcc=matthews_corrcoef(ytest,pred_naiv2)

# Bagging
print("Bagging")
bagg=BaggingClassifier()
pred_bagg=bagg.fit(xtrain2,ytrain).predict(xtest2)
bagg_accuracy=accuracy_score(ytest,pred_bagg)
bagg_kappa=cohen_kappa_score(ytest,pred_bagg)
bagg_mcc=matthews_corrcoef(ytest,pred_bagg)

# Boosting
print("GradientBoosting")
gbm2 = GradientBoostingClassifier()
pred_gbm2 =gbm2.fit(xtrain2,ytrain).predict(xtest2)
gbm_accuracy=accuracy_score(ytest,pred_gbm2)
gbm_kappa=cohen_kappa_score(ytest,pred_gbm2)
gbm_mcc=matthews_corrcoef(ytest,pred_gbm2)

# Adaboost
print("Adaboost")
ada=AdaBoostClassifier()
pred_ada=ada.fit(xtrain2,ytrain).predict(xtest2)
ada_accuracy=accuracy_score(ytest,pred_ada)
ada_kappa=cohen_kappa_score(ytest,pred_ada)
ada_mcc=matthews_corrcoef(ytest,pred_ada)

# XGBoost
print("XGBOOST")
xgbc=XGBRFClassifier()
pred_xgbc=xgbc.fit(xtrain2,ytrain).predict(xtest2)
xgbc_accuracy=accuracy_score(ytest,pred_xgbc)
xgbc_kappa=cohen_kappa_score(ytest,pred_xgbc)
xgbc_mcc=matthews_corrcoef(ytest,pred_xgbc)

# Voting Classifier
print("Voting Classifier")
vc=VotingClassifier(estimators=[("GBM",gbm2),("DTREE",dtree2),("Random Forest",rf2),("bagg",bagg)]) # Here as per the result model can be changed
pred_vc=vc.fit(xtrain2,ytrain).predict(xtest2)
vc_accuracy=accuracy_score(ytest,pred_vc)
vc_kappa=cohen_kappa_score(ytest,pred_vc)
vc_mcc=matthews_corrcoef(ytest,pred_vc)
print()

test_result={"KNN":(knn_accuracy,knn_kappa,knn_mcc),
             "DecisionTree":(dt_accuracy,dt_kappa,dt_mcc),
             "RandomForest":(rf_accuracy,rf_kappa,rf_mcc),
             "SVC":(svc_accuracy,svc_kappa,svc_mcc),
             "Gaussian naive":(nb_accuracy,nb_kappa,nb_mcc),
             "Bagging Classifier":(bagg_accuracy,bagg_kappa,bagg_mcc),
             "GradientBoosting":(gbm_accuracy,gbm_kappa,gbm_mcc),
             "AdaBoost Classifier":(ada_accuracy,ada_kappa,ada_mcc),
             "XGBOOST Classifier":(xgbc_accuracy,xgbc_kappa,xgbc_mcc),
             "Voting Classifier":(vc_accuracy,vc_kappa,vc_mcc)}
table1=pd.DataFrame(test_result,index=["Accuracy","Cohen-kappa","MCC"])
pd.DataFrame(test_result,index=["Accuracy","Cohen-kappa","MCC"]).T
