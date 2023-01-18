import pandas as pd
import textstat


def extract_features(text):
    # Split the text into words
    words = text.split()
    # Compute the frequency distribution of the words
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    # Extract features from the frequency distribution
    features = {
        'word_count': len(words),
        'unique_word_count': len(word_freq),
        'average_word_length': sum(len(word) for word in words) / len(words),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text)
    }
    return features


# Read the input CSV file
df = pd.read_csv('texts.csv')
# Extract features for each text
df['features'] = df['text'].apply(extract_features)
df = df.drop(columns=['text'])
# Convert the 'features' column to a DataFrame with separate columns for each feature
features_df = pd.DataFrame.from_records(df['features'].tolist(), columns=[
      'word_count', 'unique_word_count', 'average_word_length', 'flesch_kincaid_grade'])

# Remove the 'features' column
df = df.drop(columns=['features'])
# Concatenate the original DataFrame with the features DataFrame
df = pd.concat([df, features_df], axis=1)
# Write the results to a new CSV file
df.to_csv('features.csv', index=False)
