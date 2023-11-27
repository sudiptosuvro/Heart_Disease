from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 12)
model.fit(X_train, y_train)

y_pred_KNN = model.predict(X_test)
y_pred_KNN

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# Confusion Matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
# Accuracy Score
acc_KNN = accuracy_score(y_test, y_pred_KNN)
# Recall
recall_KNN = recall_score(y_test, y_pred_KNN)
# Precision
precision_KNN = precision_score(y_test, y_pred_KNN)
# f1 Score
f1_KNN = f1_score(y_test, y_pred_KNN)
# Classification Report
cr_KNN = classification_report(y_test, y_pred_KNN)

print(cm_KNN)
print(acc_KNN)
print(recall_KNN)
print(precision_KNN)
print(f1_KNN)
print(cr_KNN)
