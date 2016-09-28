# AA-TP1
Clasificador de Spam
# 1) Separar datos en fraccion para prueba y para testing. bajar jsons en subdirectorio jsons
cd jsons
python ../data_separator.py
#2) consolidar datos de train y test
python consolidate_input_set.py jsons/training_ham.json jsons/training_spam.json jsons/mail_training_set.json
python consolidate_input_set.py jsons/testing_ham.json jsons/testing_spam.json jsons/mail_testing_set.json
# Entrenar y ejecutar tests
python mail_classifier_compare_on_testing_data.py
