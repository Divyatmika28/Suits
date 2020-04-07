This dataset is composed of decisions made by the French Court of cassation and summaries of these decisions made by lawyer.

# Download the french Datset from :
wget ftp://echanges.dila.gouv.fr/CASS/Freemium_cass_global_20180315-170000.tar.gz


# Install the spacy module :

python install spacy
python -m spacy download fr

# Preprocess the data and generates the txt file 

python3 preprocessing_CASS.py --data_dir path_to_your_data (--clean_dir path_to_clean_data)

python3 preprocessing_CASS.py --data_dir input_data/20180315-170000/

# few output documents can be found in sampletestcases

