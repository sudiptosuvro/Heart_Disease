from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_depth = 5, criterion = 'entropy')
m = DT.fit(X, y)

y_pred_DT = m.predict(X_test)
y_pred_DT

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# Confusion Matrix
cm_DT = confusion_matrix(y_test, y_pred_DT)
# Accuracy Score
acc_DT = accuracy_score(y_test, y_pred_DT)
# Recall
recall_DT = recall_score(y_test, y_pred_DT)
# Precision
precision_DT = precision_score(y_test, y_pred_DT)
# f1 Score
f1_DT = f1_score(y_test, y_pred_DT)
# Classification Report
cr_DT = classification_report(y_test, y_pred_DT)

print(cm_DT)
print(acc_DT)
print(recall_DT)
print(precision_DT)
print(f1_DT)
print(cr_DT)

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

params_DT = {
"criterion" : ("gini", "entropy"),
"splitter" : ("best", "random"),
"max_depth" : (list(range(1, 20))),
"min_samples_split" : [2, 3, 4],
"min_samples_leaf" : list(range(1, 20)),
}
clf_DT = DecisionTreeClassifier(random_state = 3)
cv_DT = GridSearchCV(clf_DT, params_DT, scoring = "f1", n_jobs =- 1, verbose=1)

cv_DT.fit(X_train, y_train)
best_params_DT = cv_DT.best_params_

Bestparamters: ({'criterion': 'gini', 'max_depth': 1, 'min_samples_leaf': 2, 'min_samples_split': 3, 'splitter': 'random'})

cv_DT.best_params_
cv_DT.best_score_

# New Decision Tree
new_DT = DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_split=2, min_samples_leaf=1)
new_DT.fit(X_train, y_train)

new_y_pred_DT = new_DT.predict(X_test)
new_y_pred_DT

# New Model evaluation
# Confusion Matrix
new_cm_DT = confusion_matrix(y_test, new_y_pred_DT)
# Accuracy Score
new_acc_DT = accuracy_score(y_test, new_y_pred_DT)
# Recall
new_recall_DT = recall_score(y_test, new_y_pred_DT)
# Precision
new_precision_DT = precision_score(y_test, new_y_pred_DT)
# f1 Score
new_f1_DT = f1_score(y_test, new_y_pred_DT)
# Classification Report
new_cr_DT = classification_report(y_test, new_y_pred_DT)

print(new_cm_DT)
print(new_acc_DT)
print(new_recall_DT)
print(new_precision_DT)
print(new_f1_DT)
print(new_cr_DT)
