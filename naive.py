from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_NB = gnb.predict(X_test)

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# Confusion Matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)
# Accuracy Score
acc_NB = accuracy_score(y_test, y_pred_NB)
# Recall
recall_NB = recall_score(y_test, y_pred_NB)
# Precision
precision_NB = precision_score(y_test, y_pred_NB)
# f1 Score
f1_NB = f1_score(y_test, y_pred_NB)
# Classification Report
cr_NB = classification_report(y_test, y_pred_NB)

print(cm_NB)
print(acc_NB)
print(recall_NB)
print(precision_NB)
print(f1_NB)
print(cr_NB)
