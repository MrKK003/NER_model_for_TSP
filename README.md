This program is an approach to solve common salesman problem using Ukrainian news sources as an input data for calculating weights (here minutes) of every given route.

DIRECTORIES:

"wordrep_for_mitie" - contains data for word repository feature, used to compile "total_word_feature_extractor.dat", witch than used to train NER model

"train_data" - contains data for training NER model, data originally from ner-uk website, consists of annotated data from different news sources in new-uk markup.

"workspace" - directory where trained NER model is saved ("mitie_ner_model_ver1.dat")

"data" -  directory where all input data from salesman is located ("articles.xlxs" and "news_data.txt" will be compiled by main program. Outdated examples of these files are already in directory to quickly compile main program, but you can delete them and make new one)

FILES:

"total_word_feature_extractor.dat" - total word repository feature for mitie library, then to use to compile NER model 

"ner_training.py" - training NER model. Using "ner_utils_func.py" for some handy functions to help convert annotated data from ner-uk markup to mitie markup

"ner_utils_func.py" - helping converting data from different markups. (originally from  ner-uk github)

"dev-test-split.txt" - reference file with train/test split data file names from "train_data" directory

"workspace/mitie/mitie_ner_model_ver1.dat" - named entity recognision mode.

"main.py" - main program code, will work only if "workspace/mitie/mitie_ner_model_ver1.dat" and "data/routes.txt" is present 

"data/routes.txt" - main input file from salesman, contains routes (with locations names) salesman will use to visit all destinations, also contains initial travel time for every route (exactly these values we then updating using data from news sources) 