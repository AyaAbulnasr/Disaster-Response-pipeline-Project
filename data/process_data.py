import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load data from CSV files, and compine
    them into one big dataframe
    --
    arguments: messages_filepath and categories_filepath where 
    data is saved
    
    outputs: the combined dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories, on='id')
    return df


def clean_data(df):
    """
    In this function, I will clean some data:
    1- Extract the value to become a separate column
    2- Use them to create new columns with categorical values 0 or 1
   
    """
    
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[1]
    category_colnames = row.str.slice(start=0,stop=-2)
    categories.columns = category_colnames
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #if any value other than 0 or 1 exist like 2 as detected in 'related' column
    #change it to 1.
    for column in categories:
        Num = (categories[column].values == 2)
        #print(Num)
        N = [i for i, x in enumerate(Num) if x]
        #[N] = 1
        categories[column].iloc[N] = 1
        
    df.drop('categories', axis='columns',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df,categories), axis=1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    
    """
    Save cleaned dataframe
    
    arguments:
        Cleaned final dataframe 'df' which is messages and categorries data
        database_filename: database file (.db)
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Data_DisasterResponse', engine, if_exists='replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv  disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()