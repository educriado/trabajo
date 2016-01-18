# Eduardo Criado - 662844
# detector de spam

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import json
import glob
from sklearn import metrics
import random
import sys

######################################################
# Aux. functions
######################################################

# load_enron_folder: load training, validation and test sets from an enron path
def load_enron_folder(path):

   ### Load ham mails ###

   # List mails in folder
   ham_folder = path + '/ham/*.txt'
   ham_list = glob.glob(ham_folder)
   num_ham_mails = len(ham_list)

   ham_mail = []
   for i in range(0,num_ham_mails):
      ham_i_path = ham_list[i]
      print(ham_i_path)
      # Open file
      ham_i_file = open(ham_i_path, 'r')
      # Read
      ham_i_str = ham_i_file.read()
      # Convert to Unicode
      ham_i_text = ham_i_str.decode('utf-8',errors='ignore')
      # Append to the mail structure
      ham_mail.append(ham_i_text)
      # Close file
      ham_i_file.close()

   random.shuffle(ham_mail)

   # Load spam mails

   spam_folder = path + '/spam/*.txt'
   spam_list = glob.glob(spam_folder)
   num_spam_mails = len(spam_list)

   spam_mail = []
   for i in range(0,num_spam_mails):
      spam_i_path = spam_list[i]
      print(spam_i_path)
      # Open file
      spam_i_file = open(spam_i_path, 'r')
      # Read
      spam_i_str = spam_i_file.read()
      # Convert to Unicode
      spam_i_text = spam_i_str.decode('utf-8',errors='ignore')
      # Append to the mail structure
      spam_mail.append(spam_i_text)
      # Close file
      spam_i_file.close()

   random.shuffle(spam_mail)

   # Separate into training, validation and test
   num_ham_training = int(round(0.8*num_ham_mails))
   ham_training_mail = ham_mail[0:num_ham_training]
   print(num_ham_mails)
   print(num_ham_training)
   print(len(ham_training_mail))
   ham_training_labels = [0] * num_ham_training
   print(len(ham_training_labels))

   num_ham_validation = int(round(0.1*num_ham_mails))
   ham_validation_mail = ham_mail[num_ham_training:num_ham_training
                                                        +num_ham_validation]
   print(num_ham_validation)
   print(len(ham_validation_mail))
   ham_validation_labels = [0] * num_ham_validation
   print(len(ham_validation_labels))

   ham_test_mail = ham_mail[num_ham_training + num_ham_validation:num_ham_mails]
   print(num_ham_mails - num_ham_training-num_ham_validation)
   print(len(ham_test_mail))
   ham_test_labels = [0] * (num_ham_mails-num_ham_training-num_ham_validation)
   print(len(ham_test_labels))

   num_spam_training = int(round(0.8*num_spam_mails))
   spam_training_mail = spam_mail[0:num_spam_training]
   print(num_spam_mails)
   print(num_spam_training)
   print(len(spam_training_mail))
   spam_training_labels = [1] * num_spam_training
   print(len(spam_training_labels))

   num_spam_validation = int(round(0.1*num_spam_mails))
   spam_validation_mail = spam_mail[num_spam_training:num_spam_training
                                                        +num_spam_validation]
   print(num_spam_validation)
   print(len(spam_validation_mail))
   spam_validation_labels = [1] * num_spam_validation
   print(len(spam_validation_labels))

   spam_test_mail = spam_mail[num_spam_training +
                                            num_spam_validation:num_spam_mails]
   print(num_spam_mails-num_spam_training-num_spam_validation)
   print(len(spam_test_mail))
   spam_test_labels = [1] * (num_spam_mails - num_spam_training
                                                    - num_spam_validation)
   print(len(spam_test_labels))

   training_mails = ham_training_mail + spam_training_mail
   training_labels = ham_training_labels + spam_training_labels
   validation_mails = ham_validation_mail + spam_validation_mail
   validation_labels = ham_validation_labels + spam_validation_labels
   test_mails = ham_test_mail + spam_test_mail
   test_labels = ham_test_labels + spam_test_labels

   data = {'training_mails': training_mails,
           'training_labels': training_labels,
           'validation_mails': validation_mails,
           'validation_labels': validation_labels,
           'test_mails': test_mails,
           'test_labels': test_labels}

   return data


#################################################
## Devuelve el numero de errores de
## una prediccion usando una lista de comprension
#################################################
def num_errores(predicciones, labels):
   return len([(p, l) for p,l in zip(predicciones, labels) if p != l])


####################################################################
## Devuelve el mejor valor de suavizado posible
####################################################################
def mejor_alpha(k, training_mails, training_labels, validation_mails,
                                validation_labels, clasificador, uso_bigramas):
    error_mejor = len(training_mails)
    mejor_alpha = 0
    # calculamos los alphas de las particiones para los distintos correos
    tam_part_train = len(training_mails) / k
    tam_part_valid = len(validation_mails) / k
    for alpha in range(1, 10):
        print "El alpha vale ", alpha
        error_training = 0.0
        error_validation = 0.0
        for fold in range(1, k):
            # cogemos la particion para los mails de entrenamiento
            particion_training = training_mails[(tam_part_train * (fold - 1)):
                                                (tam_part_train * fold)]
            particion_training_labels = training_labels[
                                                (tam_part_train * (fold - 1)):
                                                (tam_part_train * fold)]
            # hacemos lo mismo para los de validacion
            particion_validation = validation_mails[(tam_part_valid * (fold - 1)):
                                                    (tam_part_valid * fold)]
            particion_validation_labels = validation_labels[
                                                    (tam_part_valid * (fold - 1)):
                                                    (tam_part_valid * fold)]

            # vamos a crear la bolsa de palabras y a rellenarla
            if uso_bigramas:
                cv = CountVectorizer(ngram_range=(1,2)).fit(particion_training +
                        particion_validation)
            else:
                cv = CountVectorizer().fit(particion_training +
                        particion_validation)
            matriz_training = cv.transform(particion_training)
            matriz_validation = cv.transform(particion_validation)
            # obtenemos la bolsa con la frecuencia
            tfid = TfidfTransformer().fit(matriz_training)
            frecuencias_training = tfid.transform(matriz_training)
            frecuencias_validation = tfid.transform(matriz_validation)
            # ya tenemos las bolsas de palabras con la frecuencia de aparicion
            # ahora tenemos que entrenar el clasificador
            if clasificador == "Multinomial":
                classifier = MultinomialNB(alpha).fit(frecuencias_training,
                                                    particion_training_labels)
            else:
                classifier = BernoulliNB(alpha).fit(frecuencias_training,
                                                    particion_training_labels)
            training_predictions = classifier.predict(frecuencias_training)
            validation_predictions = classifier.predict(frecuencias_validation)
            error_training += num_errores(training_predictions,
                                                    particion_training_labels)
            error_validation += num_errores(validation_predictions,
                                                   particion_validation_labels)
        # hallamos la media de errores
        error_training /= k
        error_validation /= k
        if error_validation < error_mejor:
            mejor_alpha = alpha
            error_mejor = error_validation
    return mejor_alpha

######################################################
# Main
######################################################
def main():
    print("Starting...")

    # Path to the folder containing the mails
    folder_enron1 = r'../enron-spam/enron1'
    # Load mails
    data1 = load_enron_folder(folder_enron1)
    folder_enron2 = r'../enron-spam/enron2'
    # Load mails
    data2 = load_enron_folder(folder_enron2)
    # Load mails
    folder_enron3 = r'../enron-spam/enron3'
    data3 = load_enron_folder(folder_enron3)
    # Load mails
    folder_enron4 = r'../enron-spam/enron4'
    data4 = load_enron_folder(folder_enron4)
    # Load mails
    folder_enron5 = r'../enron-spam/enron5'
    data5 = load_enron_folder(folder_enron5)
    # Load mails
    folder_enron6 = r'../enron-spam/enron6'
    data6 = load_enron_folder(folder_enron6)
    training_mails = data1['training_mails']+data2['training_mails']+data3['training_mails']+data4['training_mails']+data5['training_mails']+data6['training_mails']
    training_labels = data1['training_labels']+data2['training_labels']+data3['training_labels']+data4['training_labels']+data5['training_labels']+data6['training_labels']
    validation_mails = data1['validation_mails']+data2['validation_mails']
    validation_labels = data1['validation_labels']+data2['validation_labels']
    test_mails = data1['test_mails']+data2['test_mails']
    test_labels = data1['test_labels']+data2['test_labels']
    # ahora vamos a crear el clasificador y entrenarlo
    # primero vamos a construir los descriptores de la bolsa de palabras de cada
    # correo
    clasificador = "Bernoulli"
    uso_bigramas = False
    suavizado = mejor_alpha(5, training_mails, training_labels, validation_mails,
                                validation_labels, clasificador, uso_bigramas)
    print "El mejor suavizado es:", suavizado

    # creamos la estructura de bolsa de palabras y rellenamos
    if uso_bigramas:
        cv = CountVectorizer(ngram_range=(1,2)).fit(training_mails + test_mails)
    else:
        cv = CountVectorizer().fit(training_mails + test_mails)
    matriz_training = cv.transform(training_mails)
    matriz_test = cv.transform(test_mails)
    # vamos a normalizar
    tfid = TfidfTransformer().fit(matriz_training)
    frecuencias_training = tfid.transform(matriz_training)
    frecuencias_test = tfid.transform(matriz_test)
    if clasificador == "Multinomial":
       classifier = MultinomialNB(suavizado).fit(frecuencias_training,
                                                        training_labels)
    else:
       classifier = BernoulliNB(suavizado).fit(frecuencias_training,
                                                        training_labels)
    #Se predice con los valore de test
    test_predictions = classifier.predict(frecuencias_test)
    errores_test = num_errores(test_predictions, test_labels) / float(len(
                                                            test_predictions))
    print "Porcentaje de fallos: ", errores_test * 100, "%"
    print "Porcentaje de aciertos: ", (1 - errores_test) * 100, "%"
    # vamos a sacar las metricas
    conf_matrix = metrics.confusion_matrix(test_labels, test_predictions)
    print conf_matrix
    precision, recall, thresholds = metrics.precision_recall_curve(test_labels,
                                                            test_predictions)
    plt.plot(precision, recall, label="Curva de precision recall")
    plt.show()
    f1_score = metrics.f1_score(test_labels, test_predictions)
    print f1_score
    return 1

if __name__ == "__main__":
    main()
