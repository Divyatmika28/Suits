This dataset is composed of decisions made by the French Court of cassation and summaries of these decisions made by lawyer.
The dataset has been pulled and we followed steps from this link - https://github.com/euranova/CASS-dataset

We modified the code to get legal document in text format which has the legal text and summary.

# Download the french Datset from :


wget ftp://echanges.dila.gouv.fr/CASS/Freemium_cass_global_20180315-170000.tar.gz


# Install the spacy module :

python install spacy
python -m spacy download fr

# Preprocess the data and generates the txt file 

1. python3 preprocessing_CASS.py --data_dir path_to_your_data (--clean_dir path_to_clean_data)

Example : python3 preprocessing_CASS.py --data_dir input_data/20180315-170000/

# Few output documents can be found in sampletestcases

