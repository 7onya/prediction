import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def prepare_data(df, features_df):

    # Change all values of 2 in 1
    df['Result'] = df['Result'].replace(2, 1)
    # Merge the texts DataFrame and the train DataFrame on the 'Text A' column
    df = pd.merge(df, features_df, left_on='Text A',
                  right_on='id', how='left')

    # Rename the columns by adding _A
    df = df.rename(columns={'word_count': 'word_count_A',
                            'unique_word_count': 'unique_word_count_A',
                            'average_word_length': 'average_word_length_A',
                            'flesch_kincaid_grade': 'flesch_kincaid_grade_A'})

    # Merge the texts DataFrame and the train DataFrame on the 'Text B' column
    df = pd.merge(df, features_df, left_on='Text B',
                  right_on='id', how='left')
    # Rename the columns by adding _B
    df = df.rename(columns={'word_count': 'word_count_B',
                            'unique_word_count': 'unique_word_count_B',
                            'average_word_length': 'average_word_length_B',
                            'flesch_kincaid_grade': 'flesch_kincaid_grade_B'})

    df = df.drop(columns=['Text A', 'Text B', 'id_x', 'id_y'])
    print("Data:\n", df)
    return df


features_df = pd.read_csv('features.csv')
train_df = pd.read_csv('train.csv')
validation_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')
test_unchanged_df = pd.read_csv('test.csv')

train_df = prepare_data(train_df, features_df)
validation_df = prepare_data(validation_df, features_df)
test_df = prepare_data(test_df, features_df)


# Split the train data into X and y
y_train = train_df['Result']
X_train = train_df.drop(columns=['Result'])


# Split the validation data
y_validation = validation_df['Result']
X_validation = validation_df.drop(columns=['Result'])

# Split the test data
X_test = test_df.drop(columns=['Result'])

# Train the SVM classifier
clf = SVC()
clf.fit(X_train, y_train)
# Make predictions on the validation set
y_pred = clf.predict(X_validation)
# Compute the accuracy of the classifier
acc = accuracy_score(y_validation, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))
# Make predictions on the test set
test_unchanged_df['Result'] = clf.predict(X_test)
print(test_unchanged_df)

# Write the results to a new test CSV file
test_unchanged_df.to_csv('test_1.csv', index=False)
