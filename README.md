# Disaster Management Project on Tweet Classification

## Contents and Description:

### scripts folder
  - **process_date.py** : Reads the data from csv source files and cleans the tweets and removes duplicates. Finally, the output is saved into a MySQLdatabase.
  - **train_classifier.py** : Reads the data from MySQL db and tokenizes and lemmatizes the text. Next a model (currently random forest classifier) is trained on the data and the parameters are optimized using random search cv. The trained model is finally stored in a pickle file. 

### app folder
  - **run.py** : Houses the script that outputs the classification for a sample tweet as well as plots some related graphs. 
  - **auxiliary html scripts** : Helper html scripts for run.py
  
### data folder
  - **disaster_categories.csv** : Multiclass classifications data (labels) for input tweets.
  - **disaster_messages.csv** : Main data source containing tweets along with other info.

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.
   1. To run ETL pipeline that cleans data and stores in database
       `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   2. To run ML pipeline that trains classifier and saves
       `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. Navigate to the website *https://SPACEID-3001.SPACEDOMAIN* to view your webpage. The variables SPACEID and SPACEDOMAIN need to be replaced by values obtained from running the command *env | grep WORK* on your terminal.

## Notes:

- The scripts aren't sorted in the correct repositories and are only for convenience. To create a working directory, the original directory structure needs to be re-created. 
