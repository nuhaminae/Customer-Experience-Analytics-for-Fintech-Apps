from google_play_scraper import Sort, reviews
import csv
import pandas as pd
import logging
import os


class Scraper:
    def __init__(self, app_id, bank_name, output_folder=None):
        """
        Initialise the Scraper class with the necessary parameters.
        """
        self.app_id = app_id
        self.bank_name = bank_name
        self.output_folder = output_folder if output_folder else os.path.join(os.getcwd(), 'data') #Default to 'data' folder in current directory

        #set up logging - configured when the class is defined
        logging.basicConfig(filename='scraper.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')


    def scrape_play_store_reviews(self):
        """
        Scrape reviews from the Google Play Store for the specified bank.

        Returns the path to the CSV file containing the scraped reviews.
        """
        logging.info(f'🔄 Fetching reviews for {self.bank_name} (App ID: {self.app_id})...')

        try:
            results, _ = reviews(
                self.app_id,  #use instance variable
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=500,
                filter_score_with=None
            )

            #create output folder if it doesn't exist
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            file_name = os.path.join(self.output_folder, f'{self.bank_name}_reviews_raw.csv')

            #calculate the relative path
            current_directory = os.getcwd()

            relative_path = os.path.relpath(file_name, current_directory)

            #write to CSV without using pandas
            with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['review_text', 'rating',
                                                          'date', 'bank_name', 'source'])
                writer.writeheader()

                for entry in results:
                    writer.writerow({
                        'review_text': entry['content'],
                        'rating': entry['score'],
                        'date': entry['at'].strftime('%Y-%m-%d'),
                        'bank_name': self.bank_name, # Use instance variable
                        'source': 'Google Play'
                    })

            logging.info(f'✅ Saved {len(results)} raw reviews to {file_name}')
            print(f'✅ Saved {len(results)} raw reviews to {relative_path}')

            return file_name #return the file path


        except Exception as e:
            logging.error(f'Error occurred during scraping: {e}')
            print(f'Error occurred during scraping: {e}')
            return None #return None in case of error


    def preprocessor(self, file_name):
        """
        Preprocess the scraped reviews by removing duplicates, handling missing values,
        and normalizing the date format.
        
        Returns the path to the CSV file containing the scraped reviews.
        """
        if not file_name or not os.path.exists(file_name):
            logging.error(f'Raw CSV file not found at {file_name}. Cannot preprocess.')
            print(f'\nRaw CSV file not found at {file_name}. Cannot preprocess.')
            return None

        logging.info(f'⚙️ Preprocessing reviews from {file_name}...')

        try:
            df = pd.read_csv(file_name)

            #remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_rows:
                print(f'\nRemoved {initial_rows - len(df)} duplicate rows.')

            #handle missing values (if present)
            if df.isnull().sum().any():
                print ('Missing value found.')
                if df['review_text'].isnull().sum() > 0:
                    print ('\nMissing value found in "review_text" column. Dropping rows.')
                    df = df.dropna(subset=['review_text']) #drop rows with missing review_text
                else:
                     #handle other columns with missing values if necessary
                    print('\nHandling missing values in other columns. Filling with empty strings for text columns.')
                    for col in df.columns:
                        if df[col].dtype == 'object': 
                            df[col] = df[col].fillna('') #fill text columns with empty string
 
                print(f'\nMissing values handled. Remaining rows: {len(df)}')
            else:
                print('\nNo missing data found.')

            #normalise dates
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')

            #sort df cols
            df = df.sort_index(axis=1)

            print('\nDataFrame head:')
            print(df.head())

            print('\nDataFrame Summary:')
            print(df.info())

            print('\nDescriptive statistics for "rating" Column:')
            print(df['rating'].describe()) 

            print('\nDataFrame shape:')
            print(df.shape)

            #define processed file name and path
            processed_file_name = os.path.join(self.output_folder, f'{self.bank_name}_reviews_processed.csv')

            #calculate the relative path
            current_directory = os.getcwd()

            relative_path = os.path.relpath(file_name, current_directory)

            #save processed data to CSV
            df.to_csv(processed_file_name, index=False)
            print(f'\n✅ Saved processed reviews to {relative_path}')
            logging.info(f'✅ Saved processed reviews to {relative_path}')

            return df #return the processed DataFrame

        except Exception as e:
            logging.error(f'Error occurred during preprocessing: {e}')
            print(f'\nError occurred during preprocessing: {e}')
            return None