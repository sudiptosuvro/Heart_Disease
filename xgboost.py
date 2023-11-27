from xgboost import XGBClassifier
xgb_r = XGBClassifier()
xgb_r.fit(X_train, y_train)

y_pred_XGB = xgb_r.predict(X_test)

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# Confusion Matrix
cm_XGB = confusion_matrix(y_test, y_pred_XGB)
# Accuracy Score
acc_XGB = accuracy_score(y_test, y_pred_XGB)
# Recall
recall_XGB = recall_score(y_test, y_pred_XGB)
# Precision
precision_XGB = precision_score(y_test, y_pred_XGB)
# f1 Score
f1_XGB = f1_score(y_test, y_pred_XGB)
# Classification Report
cr_XGB = classification_report(y_test, y_pred_XGB)

print(cm_XGB)
print(acc_XGB)
print(recall_XGB)
print(precision_XGB)
print(f1_XGB)
print(cr_XGB)
