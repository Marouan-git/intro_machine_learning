library(kernlab)
library(caret)

############### choix nombre d'échantillonnages ################

n=50

set.seed(100)

################ Initialisation vecteurs ##################

vect_accuracy <- NULL
vect_sensitivity <- NULL
vect_specificity <- NULL


pt<-proc.time()
for(i in 1:n){
  
############ Initialisation paramètres ############
  
napp=1000
norm=9
out=4

########### Echantillons d'apprentissage et de test ##############

#out_app <- train[sample(which(train$y==out),50),-785]
train.1SVM=train[sample(which(train$y==norm),napp),-785]
test.1SVM=test[which(test$y==norm | test$y==out),]
truth=ifelse(test.1SVM$y==norm,"normal","abnormal")
test.1SVM=test.1SVM[,-785]

##### enlever les colonnes nulles ########
#ind=which(colMeans(train.1SVM)==0)
#train.1SVM=train.1SVM[,-ind]
#test.1SVM=test.1SVM[,-ind]

###### Entrainement du modèle ##########

model_OCC=ksvm(as.matrix(train.1SVM),type = "one-svc",kernel="rbfdot",nu=0.9, kpar="automatic",scaled=F)

######## Prédiction ############

pred=ifelse(predict(model_OCC,test.1SVM)==TRUE,"normal","abnormal")
t=table(pred,truth)

vect_accuracy <- c(vect_accuracy,confusionMatrix(t,positive = "normal")$overall[1])
vect_sensitivity <- c(vect_sensitivity,confusionMatrix(t,positive = "normal")$byClass[1])
vect_specificity <- c(vect_specificity,confusionMatrix(t,positive = "normal")$byClass[2])

}

proc.time()-pt

mean_accuracy = mean(vect_accuracy)
mean_sensitivity = mean(vect_sensitivity)
mean_specificity = mean(vect_specificity)

table_means = data.frame(mean_accuracy,mean_sensitivity,mean_specificity)

setwd("C:/Users/33650/Documents/CoursL2/MachineLearning")
write.table(table_means,"Means_OCC.csv",row.names=FALSE,dec=".", na=" ",append=FALSE)
