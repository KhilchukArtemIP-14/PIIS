import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import plot_tree
import random

if __name__=="__main__":
    pd.set_option('display.max_columns', None)

    data=pd.read_csv('data/Liver Patient Dataset (LPD)_train.csv',encoding='cp1252')
    #Some of the column names have redundant spaces
    for col in data.columns:
        data.rename(columns={col: col.strip()}, inplace=True)

    #renaming columns into more terse alternatives
    data.rename(columns={'Age of the patient': 'Age',
                         'Gender of the patient':'Gender',
                         'Alkphos Alkaline Phosphotase':'ALP',
                         'Sgpt Alamine Aminotransferase':'ALT',
                         'Sgot Aspartate Aminotransferase':'AST',
                         'Total Protiens': 'Total Proteins',
                         'ALB Albumin':'Albumin',
                         'A/G Ratio Albumin and Globulin Ratio':'A/G'},inplace=True)
    print(data)

    #removing NaN values
    msno.matrix(data)
    plt.show()

    print(f"\nTotal rows: {len(data)}\nAbsent values per column:")
    for col in data.columns:
        print(f"\t{col} - {data[col].isna().sum()}")


    print(f"\nProceeding to delete rows that contain NaN values\nRows before deletion: {len(data)}")
    data.dropna(inplace=True)
    print(f"Rows after deletion:{len(data)}")

    counts = data['Result'].value_counts()
    labels = counts.index
    values = counts.values
    plt.bar(labels, values)
    plt.xlabel('Result')
    plt.ylabel('Count')
    plt.show()

    #some encoding
    #gender
    genders={'Male':0,'Female':1}
    data['Gender']=data['Gender'].map(genders).astype(int)

    #those with 2 tend to have normal enzymes level, so they are not likely suffering from liver diseases
    #this mappping seems more logical
    results={2:0,1:1}
    data['Result']=data['Result'].map(results)

    #Determining predictors
    sns.heatmap(data.corr(),annot=True)
    plt.show()
    data.drop(columns=['Total Proteins','Age','Gender'],inplace=True)

    sns.heatmap(data.corr(),annot=True)
    plt.show()

    data['Globulin']=data['Albumin']/data['A/G']
    data.drop(columns=['Total Bilirubin','AST','A/G'],inplace=True)

    sns.heatmap(data.corr(),annot=True)
    plt.show()

    data.drop(columns=['Globulin'],inplace=True)
    print(data)


    #splitting data
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['Result'], test_size=0.2, random_state=42)



    # Create an instance of SMOTE
    oversampler = SMOTE(random_state=42)

    # Apply oversampling to the training data
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    tree_clf = DecisionTreeClassifier(max_depth=4)
    tree_clf.fit(X_train_resampled, y_train_resampled)

    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_clf, feature_names=X_train.columns,filled=True, class_names=["0", "1"])
    plt.show()

    y_pred = tree_clf.predict(X_test)
    DTC_acc = accuracy_score(y_test, y_pred)
    DTC_recall = recall_score(y_test, y_pred)
    print(f'Metrics are:\n\tAccuracy: {DTC_acc}\n\tRecall:{DTC_recall}\n\tF-score:{2 * DTC_acc * DTC_recall / (DTC_acc + DTC_recall)}')

    cm = confusion_matrix(y_test, y_pred)
    dtc_type2_err = cm[1, 0]
    dtc_type1_err = cm[0, 1]
    sns.heatmap(cm,annot=True)
    plt.grid(False)
    plt.show()



    #some experiments

    print("Performing some experiments:")
    for i in range(5):
        print(f"\nExperiment number{i+1}:")
        random_index = random.randint(0, len(X_test) - 1)

        sample_features = X_test.iloc[random_index]
        sample_label = y_test.iloc[random_index]

        print(f"\tDirect bilirubin: {sample_features['Direct Bilirubin']}")
        print(f"\tALP: {sample_features['ALP']}")
        print(f"\tALT: {sample_features['ALT']}")
        print(f"\tAlbumin: {sample_features['Albumin']}")
        print(f"\tActual result: {sample_label}")

        sample_prediction = tree_clf.predict([sample_features])[0]
        print(f"\tPredicted result: {sample_prediction}")