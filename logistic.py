from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_LR = lr.predict(X_test)

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# Confusion Matrix
cm_LR = confusion_matrix(y_test, y_pred_LR)
# Accuracy Score
acc_LR = accuracy_score(y_test, y_pred_LR)
# Recall
recall_LR = recall_score(y_test, y_pred_LR)
# Precision
precision_LR = precision_score(y_test, y_pred_LR)
# f1 Score
f1_LR = f1_score(y_test, y_pred_LR)
# Classification Report
cr_LR = classification_report(y_test, y_pred_LR)

print(cm_LR)
print(acc_LR)
print(recall_LR)
print(precision_LR)
print(f1_LR)
print(cr_LR)
