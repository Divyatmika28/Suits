# LetSum

## Dataset

- Download the CASS dataset named Freemium_cass_global_20180315-170000.tar.gz from [ftp://echanges.dila.gouv.fr/CASS/](ftp://echanges.dila.gouv.fr/CASS/)  

### main.py

Main for segmenting the data. It writes the segmented data so it can be used for selection.

### formulated_constants_french.py

Contains a dictionary of the key phrases for each theme. (Introduction, Context, Analysis, Conclusion). These key phrases were from the LetSum paper and created by create_key_phrases.py

### lets_french.py

Segments the data by using key phrases from formulated_constants_french and the number of time a phrase appears in a paragraph. Segments the data in the order of Introduction, Context, Analysis, Conclusion. There is also a constraint on how many paragraphs a theme could have. 

