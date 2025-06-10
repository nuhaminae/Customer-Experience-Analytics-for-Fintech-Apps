#import 
import pandas as pd
import os
import random
import string

df_path = os.path.join(os.path.dirname(os.getcwd()), 'data/all_preprocessed_reviews.csv')

def process_df_for_db (df_path): 

    #read df
    df = pd.read_csv(df_path)
    
    #change 'date' column to 'review_date' tp avoid conflict with database nomenclature 
    df = df.rename(columns={'date': 'review_date'})

    #assign unique bank id to each banks
    df['bank_id'] = '' #initalise 'bank_id' column
    df.loc[df['bank_name'] == 'Abyssinia', 'bank_id'] = '1'
    df.loc[df['bank_name'] == 'Commercial', 'bank_id'] = '2'
    df.loc[df['bank_name'] == 'Dashen', 'bank_id'] = '3'

    #add a unique review_id to each row
    def generate_random_string(length):
        """Generate a random string of fixed length."""
        digits =  string.digits
        return ''.join(random.choice(digits) for i in range(length))

    df['review_id'] = [generate_random_string(6) for _ in range(len(df))] #generate unique IDs

    #sort df columns
    df = df[['review_id','bank_id', 'bank_name',
            'rating',	'review_date',
            'review_text', 'source']]
    
    return df #return processed df