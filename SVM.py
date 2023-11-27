from sklearn.svm import SVC
svclassifier = SVC(kernel = 'rbf', C = 30, gamma = 'auto')
svclassifier.fit(X_train, y_train)

y_predSVM = svclassifier.predict(X_test)

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# Confusion Matrix
cm_SVM = confusion_matrix(y_test, y_predSVM)
# Accuracy Score
acc_SVM = accuracy_score(y_test, y_predSVM)
# Recall
recall_SVM = recall_score(y_test, y_predSVM)
# Precision
precision_SVM = precision_score(y_test, y_predSVM)
# f1 Score
f1_SVM = f1_score(y_test, y_predSVM)
# Classification Report
cr_SVM = classification_report(y_test, y_predSVM)

print(cm_SVM)
print(acc_SVM)
print(recall_SVM)
print(precision_SVM)
print(f1_SVM)
print(cr_SVM)
