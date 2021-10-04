################ Détails sur les datasets #####################

# Le training set comporte 60000 exemplaires de chiffres (0 à 9) écrits à la main
# Le test set en comporte 10000

# Les labels correspondent donc aux chiffres de 0 à 9  (= 10 catégories)

# La taille des chiffres a été normalisée, les chiffres ont été centrés dans une image dont la taille a été normalisée (28*28 pixels)

# Les images ont déjà été mises en niveaux de gris

# Ils ont bien fait attention à ce que les personnes ayant écrit les chiffres du training dataset soient différentes de celles du test dataset





################ Chargement des datasets ######################

# Les fichiers nécessitaient un traitement pour être lus sous R, j'ai récupéré ce code à l'adresse suivante : https://gist.github.com/daviddalpiaz/ae62ae5ccd0bada4b9acd6dbc9008706
# J'ai simplement modifié le nom des variables contenant les datasets : train, test, train$y et test$y


# modification of https://gist.github.com/brendano/39760
# automatically obtains data from the web
# creates two data frames, test and train
# labels are stored in the y variables of each data frame
# can easily train many models using formula `y ~ .` syntax

# download data from http://yann.lecun.com/exdb/mnist/
download.file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
              "train-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
              "train-labels-idx1-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
              "t10k-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
              "t10k-labels-idx1-ubyte.gz")

install.packages('R.utils')

# gunzip the files
R.utils::gunzip("train-images-idx3-ubyte.gz")
R.utils::gunzip("train-labels-idx1-ubyte.gz")
R.utils::gunzip("t10k-images-idx3-ubyte.gz")
R.utils::gunzip("t10k-labels-idx1-ubyte.gz")



# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images
train = load_image_file("train-images-idx3-ubyte") # images en ligne et pixels en colonne
test  = load_image_file("t10k-images-idx3-ubyte")

# load labels
train$y = as.factor(load_label_file("train-labels-idx1-ubyte"))  # on définit ici les étiquettes comme des niveaux pour que le modèle les considère comme des catégories et non pas des scores
test$y  = as.factor(load_label_file("t10k-labels-idx1-ubyte"))   # vecteurs comportant les labels

# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

# view test image
show_digit(train[14,])
show_digit(test[500,])


summary(test[ , 0:100])  # distribution des valeurs pour les pixels 
summary(train[ ,0:100]) # de l'image 28*28 (784 colonnes), ici pixels de 0 à 100

summary(test$y) /10000  # on peut voir le nombre d'observation par catégorie
summary(train$y)/60000  # en fréquence




######################### Modèle SVM ###############################





install.packages('e1071')  
install.packages('kernlab')
install.packages('caret')

library(kernlab) # contient ksvm
library(caret)  # contient confusion matrix
library(e1071)
set.seed(100) # pour fixer la graine du générateur aléatoire
# et obtenir les mêmes résultats avec sample par exemple d'une session à l'autre ou même d'un ordi à l'autre

# échantillon car temps de traitement trop long avec toutes les données

sample_indices <- sample(1: nrow(train), 5000) # extracting subset of 5000 samples for modelling
train <- train[sample_indices, ]
