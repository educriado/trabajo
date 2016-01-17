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
   ham_folder = path + '\ham\*.txt'
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

   spam_folder = path + '\spam\*.txt'
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
   ham_training_labels = [0]*num_ham_training
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


######################################################
# Main
######################################################
def main(argv):
    print("Starting...")

    # Path to the folder containing the mails
    folder_enron1 = r'../enron-spam/enron1' 
    # Load mails
    data1 = load_enron_folder(folder_enron1)
    folder_enron2 = r'../enron-spam/enron2'
    # Load mails
    data2 = load_enron_folder(folder_enron2)
    training_mails = data1['training_mails']+data2['training_mails']
    training_labels = data1['training_labels']+data2['training_labels']
    validation_mails = data1['validation_mails']+data2['validation_mails']
    validation_labels = data1['validation_labels']+data2['validation_labels']
    test_mails = data1['test_mails']+data2['test_mails']
    test_labels = data1['test_labels']+data2['test_labels']
    # ahora vamos a crear el clasificador y entrenarlo
    # primero vamos a construir los descriptores de la bolsa de palabras de cada
    # correo
    if argv == "bernoulli":
        # llamada a la funcion que calcule siguiendo este metodo
        return 1
    elif argv == "multinomial":
        # llamada a esta otra
        return 2
    else:
        # error
        return -1

    cv = CountVectorizer()
    cv.fit(training_mails)
    # guardamos la matriz
    bolsa = cv.tranform(training_mails)
    return 1

if __name__ == "__main__":
    main(argv)
