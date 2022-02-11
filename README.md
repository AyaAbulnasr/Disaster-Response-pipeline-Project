# Disaster Response Pipeline Project

### Introducton
#### Project Describtion:
In this Project, I analyzed the attached datasets file contains tweet and messages a real life disaster responses. The aim of the project is to build a Natural Language Processing tool or API that classifies the recieved messages as the following  sample screenshot.
![image](https://user-images.githubusercontent.com/80397129/153633555-7c6ee995-e6a4-42d8-a301-d67cc3503d37.png)

### Preprocessing
I had a preprocessing statge which found at data/process_data.py, it's containing an ETL pipeline to do the following:

1. Reading data from the csv files disaster_messages.csv and disaster_categories.csv.
2. Both the messages and the categories datasets are merged.
3. Cleaning merged dataframe .
4. Duplicated mesages are removed.
5. storeing cleaned data over data/DisasterResponse.db.
6. 
### Machine Learning Pipeline
ML pipeline is implemented in models/train_classifier.py.

1. Exort the data from data/DisasterResponse.db.
2. Splitting dataframe trainging and testing sets.
3. A function tokenize() is implemented to clean the messages data and tokenize it for tf-idfcalculations.
4. Pipelines are implemented for text and machine learning processing.
5. Parameter selection is based on GridSearchCV.
6. Trained classifier is stored in models/classifier.pkl.

### Flask App
Flask app is implemented in the app folder.
Main page gives data overview as shown in the attached images. Main target is to leave the message the the msg box and it will categorize the message in its genre.

### Data Overview:

MSG_Genre_Distribution.png




### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
