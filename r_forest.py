from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 100)
RF.fit(X_train, y_train)

y_pred_RF = RF.predict(X_test)
y_pred_RF

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# Confusion Matrix
cm_RF = confusion_matrix(y_test, y_pred_RF)
# Accuracy Score
acc_RF = accuracy_score(y_test, y_pred_RF)
# Recall
recall_RF = recall_score(y_test, y_pred_RF)
# Precision
precision_RF = precision_score(y_test, y_pred_RF)
# f1 Score
f1_RF = f1_score(y_test, y_pred_RF)
# Classification Report
cr_RF = classification_report(y_test, y_pred_RF)

print(cm_RF)
print(acc_RF)
print(recall_RF)
print(precision_RF)
print(f1_RF)
print(cr_RF)

# Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
max_features': ['auto', 'sqrt'],
max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
min_samples_split': [2, 5, 10],
min_samples_leaf': [1, 2, 4],
bootstrap': [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
'max_depth': max_depth, 'min_samples_split': min_samples_split
'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap
clf_RF = RandomForestClassifier(random_state = 42)
cv_RF = RandomizedSearchCV( estimator=clf_RF, scoring='f1', param_distributions=param_distributions, verbose=2, random_state=42, n_jobs=-1, n_iter=10, cv=5)

cv_RF.fit(X_train, y_train)
best_params_RF = cv_RF.best_params_
print(f"Best paramters: {best_params_RF}")

clf_RF = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=1, max_features='auto')
clf_RF.fit(X_train, y_train)

new_y_pred_RF = clf_RF.predict(X_test)
new_y_pred_RF

# Confusion Matrix
new_cm_RF = confusion_matrix(y_test, new_y_pred_RF)
# Accuracy Score
new_acc_RF = accuracy_score(y_test, new_y_pred_RF)
# Recall
new_recall_RF = recall_score(y_test, new_y_pred_RF)
# Precision
new_precision_RF = precision_score(y_test, new_y_pred_RF)
# f1 Score
new_f1_RF = f1_score(y_test, new_y_pred_RF)
# Classification Report
new_cr_RF = classification_report(y_test, new_y_pred_RF)

print(new_cm_RF)
print(new_acc_RF)
print(new_recall_RF)
print(new_precision_RF)
print(new_f1_RF)
print(new_cr_RF)             
