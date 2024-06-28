import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

estimators = 100

import warnings
warnings.filterwarnings('ignore')

def proc_file(df):
    df = df.drop(['Timestamp', 'Data'], axis=1, errors = 'ignore')
    return df

path = os.getcwd()

# file_name = input("write your train file name: ")
train_file = os.path.join(path, "source", "CHCD", "Pre_train_total_proc.csv")
test_file = os.path.join(path, "source", "CHCD", "Pre_submit_total_proc.csv")
final_test_file = os.path.join(path,"source", "CHCD", "Fin_host_session_submit_total_proc.csv")


print("start read files")
df = pd.read_csv(train_file)
tf = pd.read_csv(test_file)
ftf = pd.read_csv(final_test_file)

print("start processing data")
df = proc_file(df)
tf = proc_file(tf)
ftf = proc_file(ftf)

print(df.info())

feature_columns = list(df.columns.difference(['Class','SubClass']))

train_x = df[feature_columns]
test_x = tf[feature_columns]
final_test_x = ftf[feature_columns]

all_class = ['Normal', 'Attack']
all_subclass = ['Normal', 'Flooding', 'Spoofing', 'Replay', 'Fuzzing']

train_y_C = df['Class']
test_y_C = tf['Class']
final_test_y_C = ftf['Class']
depth_C = 30
estimators_C = 190

train_y_S = df['SubClass']
test_y_S = tf['SubClass']
final_test_y_S = ftf['SubClass']
depth_S = 30
estimators_S = 105

what = input("train or use model(t/m): ")
if  what == 't':
    print(train_x.shape)
    print(train_y_C.shape)
    print(train_y_S.shape)

    clf_C = RandomForestClassifier(n_estimators=estimators_C,max_depth=depth_C,n_jobs = -1,verbose = 2)
    print("start training model C")
    clf_C.fit(train_x,train_y_C)
    
    clf_S = RandomForestClassifier(n_estimators=estimators_S,max_depth=depth_S,n_jobs = -1,verbose = 2)
    print("start training model S")
    clf_S.fit(train_x,train_y_S)
    
    print("start record model")
    model_pkl_file_C = os.path.join(path, f"Pre_train_total_model_C.pkl")
    with open(model_pkl_file_C, 'wb') as file:
        pickle.dump(clf_C, file)
    print(model_pkl_file_C)

    model_pkl_file_S = os.path.join(path, f"Pre_train_total_model_S.pkl")
    with open(model_pkl_file_S, 'wb') as file:
        pickle.dump(clf_S, file)
    print(model_pkl_file_S)
    print("end record model")
else:
    model_pkl_file_C = os.path.join(path, f"Pre_train_total_model_C.pkl")
    model_pkl_file_S = os.path.join(path, f"Pre_train_total_model_S.pkl")
    print("load model")

    with open(model_pkl_file_C, 'rb') as file:
        clf_C = pickle.load(file)

    with open(model_pkl_file_S, 'rb') as file:
        clf_S = pickle.load(file)

start_C = time.time()
predict_C = clf_C.predict(test_x)
end_C = time.time()

start_S = time.time()
predict_S = clf_S.predict(test_x)
end_S = time.time()

start_C = time.time()
final_predict_C = clf_C.predict(final_test_x)
end_C = time.time()

start_S = time.time()
final_predict_S = clf_S.predict(final_test_x)
end_S = time.time()

C_label = tf['Class'].map({0: '0.Normal', 1: '1.Attack' })
S_label = tf['SubClass'].map({0: '0.Normal', 1: '1.Flooding', 2: '2.Spoofing', 3: '3.Replay', 4: '4.Fuzzing'})

final_C_label = ftf['Class'].map({0: '0.Normal', 1: '1.Attack' })
final_S_label = ftf['SubClass'].map({0: '0.Normal', 1: '1.Flooding', 2: '2.Spoofing', 3: '3.Replay', 4: '4.Fuzzing'})

predict_C_labels = pd.Series(predict_C).map({0: '0.Normal', 1: '1.Attack' })
predict_S_labels = pd.Series(predict_S).map({0: '0.Normal', 1: '1.Flooding', 2: '2.Spoofing', 3: '3.Replay', 4: '4.Fuzzing'})

final_predict_C_labels = pd.Series(final_predict_C).map({0: '0.Normal', 1: '1.Attack' })
final_predict_S_labels = pd.Series(final_predict_S).map({0: '0.Normal', 1: '1.Flooding', 2: '2.Spoofing', 3: '3.Replay', 4: '4.Fuzzing'})


report_name = input("write your report name: ")
print(f"Class Accuracy: {accuracy_score(C_label, predict_C_labels)}\nPredict time per sample: {(end_C - start_C) / tf.shape[0]:.10f}")
with open(f"{report_name}_Binary_report.txt",'w') as text_file:
    print(classification_report(C_label, predict_C_labels, digits = 4), file = text_file)

print(f"SubClass Accuracy: {accuracy_score(S_label, predict_S_labels)}\nPredict time per sample: {(end_S - start_S) / tf.shape[0]:.10f}")
with open(f"{report_name}_Multi_report.txt",'w') as text_file :
    print(classification_report(S_label, predict_S_labels, digits = 4), file = text_file)

print(f"final Class Accuracy: {accuracy_score(final_C_label, final_predict_C_labels)}\nPredict time per sample: {(end_C - start_C) / tf.shape[0]:.10f}")
with open(f"{report_name}_Binary_Final_report.txt",'w') as text_file:
    print(classification_report(final_C_label, final_predict_C_labels, digits = 4), file = text_file)

print(f"final SubClass Accuracy: {accuracy_score(final_S_label, final_predict_S_labels)}\nPredict time per sample: {(end_S - start_S) / tf.shape[0]:.10f}")
with open(f"{report_name}_Multi_Final_report.txt",'w') as text_file:
    print(classification_report(final_S_label, final_predict_S_labels, digits = 4), file = text_file)

# Plotting feature importances and confusion matrices
plt.figure(figsize=(16, 12))

# Feature importances for Class
ftr_importances_values_C = clf_C.feature_importances_
ftr_importances_C = pd.Series(ftr_importances_values_C, index=train_x.columns)
ftr_top_C = ftr_importances_C.sort_values(ascending=False)

plt.subplot(231)
plt.title('Feature Importances for Class')
sns.barplot(x=ftr_top_C, y=ftr_top_C.index)
final_cf_matrix_C = confusion_matrix(final_C_label, final_predict_C_labels)

plt.subplot(232)
sns.heatmap(final_cf_matrix_C, annot=True, fmt='d', cmap='Blues', xticklabels=all_class, yticklabels=all_class)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Class')

# Confusion matrix for Class
cf_matrix_C = confusion_matrix(C_label, predict_C_labels)

plt.subplot(233)
sns.heatmap(cf_matrix_C, annot=True, fmt='d', cmap='Blues', xticklabels=all_class, yticklabels=all_class)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Class')

# Feature importances for SubClass
ftr_importances_values_S = clf_S.feature_importances_
ftr_importances_S = pd.Series(ftr_importances_values_S, index=train_x.columns)
ftr_top_S = ftr_importances_S.sort_values(ascending=False)

plt.subplot(234)
plt.title('Feature Importances for SubClass')
sns.barplot(x=ftr_top_S, y=ftr_top_S.index)
final_cf_matrix_S = confusion_matrix(final_S_label, final_predict_S_labels)

plt.subplot(235)
sns.heatmap(final_cf_matrix_S, annot=True, fmt='d', cmap='Blues', xticklabels=all_subclass, yticklabels=all_subclass)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for SubClass')

# Confusion matrix for SubClass
cf_matrix_S = confusion_matrix(S_label, predict_S_labels)

plt.subplot(236)
sns.heatmap(cf_matrix_S, annot=True, fmt='d', cmap='Blues', xticklabels=all_subclass, yticklabels=all_subclass)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for SubClass')

plt.tight_layout()
plt.show()

# Sample to reduce plotting time and complexity
# sample_df = ftf.sample(5000)  
# sns.pairplot(sample_df[feature_columns + ['SubClass']], palette= 'coolwarm', hue='SubClass')
# plt.suptitle('Pairplot of Features', y=1.02)
# plt.show()

# sample_df = df.sample(5000)  
# sns.pairplot(sample_df[feature_columns + ['SubClass']], palette= 'coolwarm', hue='SubClass')
# plt.suptitle('Pairplot of Features', y=1.02)
# plt.show()

# df_sample = df.sample(5000)
# tf_sample = ftf.sample(5000)

# # Feature columns
# columns = list(df.columns.difference(['Class', 'SubClass', 'ID_Frequency']))

# # Create subplots
# num_features = len(columns)

# fig1, axes1 = plt.subplots(nrows=num_features, ncols=1, figsize=(10, num_features * 4))
# for ax, feature in zip(axes1, columns):
#     sns.scatterplot(data=df_sample, x='ID_Frequency', y=feature, hue='SubClass', palette='coolwarm', ax=ax, marker='o', alpha=0.5)
#     ax.legend(loc = 'upper right', bbox_to_anchor = (1,1))

# fig1.suptitle('Train Data: Comparison of ID_Frequency with Other Features', y=1.02, fontsize=16)
# fig1.tight_layout()

# # Plot for tf (Test Data)
# fig2, axes2 = plt.subplots(nrows=num_features, ncols=1, figsize=(10, num_features * 4))
# for ax, feature in zip(axes2, columns):
#     sns.scatterplot(data=tf_sample, x='ID_Frequency', y=feature, hue='SubClass', palette='viridis', ax=ax, marker='o', alpha=0.5)
#     ax.legend(loc = 'upper right', bbox_to_anchor = (1,1))

# fig2.suptitle('Test Data: Comparison of ID_Frequency with Other Features', y=1.02, fontsize=16)
# fig2.tight_layout()

# plt.show()

# Heatmap of correlation matrix
# corr_matrix = df[feature_columns].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2g')
# plt.title('Correlation Matrix of Features')
# plt.show()