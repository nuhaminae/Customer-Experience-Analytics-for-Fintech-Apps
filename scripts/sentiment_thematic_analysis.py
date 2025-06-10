#important python lib and module
import pandas as pd
import os

#for visualisation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#for sentiment analysis
from transformers import pipeline

#for text modelling
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


def save_plot(plot_folder, plot_name, plot_path):
    """
    Saves the current matplotlib plot to a specified location.

    Args:
        plot_folder (str): The folder to save the plot.
        plot_name (str): The name of the plot file.
        plot_path (str): The full path to save the plot.
    """
    
    #create the directory if it doesn't exist
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    #save plot
    plt.savefig(plot_path)

    #calculate the relative path
    current_directory = os.getcwd()
    relative_plot_path = os.path.relpath(plot_path, current_directory)

    #display message
    print(f'\nPlot is saved to {relative_plot_path}.\n')

#load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english")

#download necessary NLTK data
nltk.download('punkt', quiet=True)              #splits text
nltk.download('stopwords', quiet=True)          #removes common words like 'a', 'is', 'the', 'in'
nltk.download('wordnet', quiet=True)            #identifies synonyms

#load spacy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    #print("Downloading spacy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Preprocesses text by lowercasing, removing non-alphanumeric chars,
    tokenizing, removing stop words, and lemmatizing.

    Args:
        text (str): The input text.

    Returns:
        str: The processed text.
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        #remove punctuation and stop words, and keep only alphabetic tokens
        if not token.is_punct and not token.is_stop and token.is_alpha:
            tokens.append(token.lemma_)  #use lemmatization
    return " ".join(tokens)


def calculate_tfidf(text_data, max_features=1000, ngram_range=(1, 2)): #considerd unigrams and bigrams
    """
    Calculates TF-IDF scores for a list of text documents.

    Args:
        text_data (list): A list of processed text documents.
        max_features (int): Maximum number of features for TF-IDF.
        ngram_range (tuple): Range of n-grams to consider.

    Returns:
        tuple: tfidf_matrix, feature_names
    """
    #calculate Term Frequency-Inverse Document Frequency (TF-IDF) to identify important terms
    #initalise tfidf_vectorizer for upto 1000 single words and two-word phrases 
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    #calculate tfidf_matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

    #get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def extract_keywords_spacy(text):
    """
    Extracts potential keywords from text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        str: A comma-separated string of extracted keywords.
    """
    doc = nlp(text)
    keywords = []
    for token in doc:
        #part of speech can be customised 
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV', 'VERB'] and not token.is_stop and not token.is_punct:
            keywords.append(token.lemma_)
    return ", ".join(keywords)

def assign_theme(review_text, keywords):
    """
    Assigns a theme to a review based on keywords and review content.

    Args:
        review_text (str): The full review text.
        keywords (str): A comma-separated string of keywords extracted from the review.

    Returns:
        str: The identified theme for the review.
    """

    #define rules based on keywords and review content
    review_text_lower = review_text.lower()
    if any(word in review_text_lower for word in ['login', 'account', 'access', 'password']):
        return 'Account Access Issues'
    elif any(word in review_text_lower for word in ['transfer', 'transaction', 'slow', 'speed', 'delay', 'payment']):
        return 'Transaction Performance/Payment Issues'
    elif any(word in review_text_lower for word in ['support', 'customer service', 'help', 'agent', 'contact']):
        return 'Customer Support'
    elif any(word in review_text_lower for word in ['ui', 'interface', 'design', 'easy', 'user', 'navigation', 'app']):
        return 'User Interface & Experience'
    elif any(word in review_text_lower for word in ['bug', 'error', 'crash', 'freeze', 'issue', 'fix']):
        return 'Bugs and Errors'
    elif any(word in review_text_lower for word in ['update', 'version', 'install', 'download']):
        return 'App Updates/Installation'
    elif any(word in review_text_lower for word in ['security', 'secure', 'fraud', 'safe', 'otp']):
        return 'Security Concerns'
    elif any(word in review_text_lower for word in ['feature', 'request', 'new', 'add', 'missing']):
        return 'Feature Requests/Missing Features'
    elif any(word in review_text_lower for word in ['notification', 'alert']):
        return 'Notifications'
    elif any(word in review_text_lower for word in ['performance', 'speed', 'lag']):
        return 'App Performance'
    else:
        return 'Other'

def dfloader_and_analyser (df_path, bank_name, output_folder, plot_folder):
    """
    Loads a DataFrame, performs sentiment analysis, text preprocessing,
    keyword extraction, theme assignment, and saves the processed DataFrame.

    Args:
        df_path (str): The path to the input CSV file.
        bank_name (str): The name of the bank to use in the output file name.
        output_folder (str): The path to the folder where the output CSV will be saved.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """

    df=pd.read_csv(df_path)

    #compute sentiment score 
    df['sentiment_score'] = df['review_text'].apply(lambda x: sentiment_pipeline(x)[0])
    df[['sentiment','score']] = pd.DataFrame(df['sentiment_score'].tolist(), index= df.index)
   
    #sentiment distribution plot
    print ('Sentiment Value Counts Plot:')
    df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title(f'{bank_name} - Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0) 

    #select plot directory and plot name to save plot
    plot_name = f'{bank_name} - Sentiment Distribution.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)

    #show plot
    plt.show()
    #close plot to free up space
    plt.close()
    
    #group by bank and rating, and calculate the mean sentiment score and count
    aggregated_sentiment = df.groupby(['bank_name', 
                                       'rating']).agg(mean_sentiment_score=('score', 
                                                                            'mean'),
                                                                            rating_count=('rating', 
                                                                                          'count')).reset_index()
    print('\nAggregated Results:')
    print(aggregated_sentiment)

    #app rating distribution plot
    print ('Rating Distribution Plot:\n')
    aggregated_sentiment.plot( x= 'rating', y='rating_count',
                              kind='bar', color=['blue','green', 'yellow','red', 'black'], legend=False)
    plt.title(f'{bank_name} - Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=0) 

    #select plot directory and plot name to save plot
    plot_name = f'{bank_name} - Rating Distribution.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)

    #show plot
    plt.show()
    #close plot to free up space
    plt.close()
    
    #work extraction from review text
    df['processed_review'] = df['review_text'].apply(preprocess_text)

    #vectorize the dataset
    tfidf_matrix, feature_names = calculate_tfidf(df['processed_review'])
    print('\nTF-IDF Matrix Shape:', tfidf_matrix.shape)
    print('First 25 Feature Names:', feature_names[:25])

    #filter positive and negative reviews for keyword extraction
    positive_reviews = df[df['sentiment'] == 'POSITIVE']['processed_review']
    negative_reviews = df[df['sentiment'] == 'NEGATIVE']['processed_review']

    #extract keywords from positive reviews
    #ensure there are positive reviews to process
    if not positive_reviews.empty:
        vectorizer_pos = TfidfVectorizer(max_features=20)
        X_pos = vectorizer_pos.fit_transform(positive_reviews.tolist())
        print('\nTop 20 Keywords in Positive Reviews:', 
              vectorizer_pos.get_feature_names_out())
    else:
        print('\nNo positive reviews to extract keywords from.')

    #extract keywords from negative reviews
    #ensure there are negative reviews to process
    if not negative_reviews.empty:
        vectorizer_neg = TfidfVectorizer(max_features=20)
        X_neg = vectorizer_neg.fit_transform(negative_reviews.tolist())
        print('\n Top 20 Keywords in Negative Reviews:', 
              vectorizer_neg.get_feature_names_out())
    else:
        print('\nNo negative reviews to extract keywords from.')

    #word cloud for positive reviews
    positive_text = ' '.join(positive_reviews)
    wordcloud = WordCloud(width=800, 
                          height=400, 
                          background_color='white').generate(positive_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{bank_name} - Word Cloud for Positive Reviews')
    
    #select plot directory and plot name to save plot
    plot_name = f'{bank_name} -Word Cloud for Positive Reviews.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)

    #show plot
    plt.show()
    #close plot to free up space
    plt.close()

    #word cloud for negative reviews
    negative_reviews = ' '.join(negative_reviews)
    wordcloud = WordCloud(width=800, 
                          height=400, 
                          background_color='white').generate(negative_reviews)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{bank_name} - Word Cloud for Negative Reviews')

    #select plot directory and plot name to save plot
    plot_name = f'{bank_name} - Word Cloud for Negative Reviews.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    
    #show plot
    plt.show()
    #close plot to free up space
    plt.close()
    
    #apply spaCy keyword extraction
    df['spacy_keywords'] = df['review_text'].apply(extract_keywords_spacy)

    #apply the theme assignment function
    df['identified_theme(s)'] = df.apply(lambda row: assign_theme(row['review_text'], 
                                                                  row['spacy_keywords']),
                                                                  axis=1)

    result_df = df[['date','rating','review_text', 'sentiment',
                    'score', 'spacy_keywords', 'identified_theme(s)', 'bank_name']]

     
    #define processed file name and path
    results_df_name = os.path.join(output_folder, f'{bank_name}_result.csv')

    #create output folder if it doesn't exist
    if not os.path.exists(output_folder):
                os.makedirs(output_folder)

    #calculate the relative path
    current_directory = os.getcwd()

    relative_path = os.path.relpath(results_df_name, current_directory)

    #save processed data to CSV
    df.to_csv(results_df_name, index=False)
    
    print('\nProcessed DataFrame Saved to:', relative_path)
    
    print('\nDataFrame Head:')
    print(result_df.head())

    #plot reviews theme barplot 
    #set fig size
    plt.figure(figsize=(10, 6))

    df['identified_theme(s)'].value_counts().plot(kind='bar', color=['skyblue',
                                                                    'lightcoral', 
                                                                    'lightgreen', 
                                                                    'gold', 
                                                                    'plum'])
    plt.title(f'{bank_name} - Value Count of User Review Themes')
    plt.xlabel('Review Theme')
    plt.ylabel('Count')
    plt.xticks(rotation=45,
            ha='right'); #semicolon to suppress the array of value count output
    plt.tight_layout()

    #select plot directory and plot name to save plot
    plot_name = f'{bank_name} - Value Count of User Review Themes.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    
    #show plot
    plt.show()
    #close plot to free up space
    plt.close()

    return df

#merge dfs for ease database access
def concat_and_save_dfs(df_paths, df_folder, df_name):
    """
    Reads multiple CSV files into DataFrames, concatenates them,
    and saves the concatenated DataFrame to a new CSV file.

    Args:
        df_paths (list): A list of strings, where each string is the path to a CSV file.
        df_folder (str): The path to the folder where the output CSV will be saved.
        df_name (str): The name of the output CSV file.

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """

    dataframes = []
    for df_path in df_paths:
        df = pd.read_csv(df_path)
        dataframes.append(df)

    #concatenate the DataFrames
    preprocessed_df = pd.concat(dataframes, ignore_index=True)

    #create output folder if it doesn't exist
    if not os.path.exists(df_folder):
        os.makedirs(df_folder)

    #create the full path for the output file
    df_filepath = os.path.join(df_folder, df_name)

    #save the concatenated DataFrame to CSV
    preprocessed_df.to_csv(df_filepath, index=False)

    #calculate the relative path
    current_directory = os.getcwd()
    relative_df_path = os.path.relpath(df_filepath, current_directory)

    print(f'Concatenated DataFrame saved to: {relative_df_path}')

    return preprocessed_df