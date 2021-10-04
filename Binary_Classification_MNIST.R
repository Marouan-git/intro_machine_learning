library(class)
library(kernlab)
library(caret)

############### choix nombre d'échantillonnages ################

n=50

set.seed(100)

################ Initialisation vecteurs ##################

vect_accuracy_svm <- NULL
vect_sensitivity_svm <- NULL
vect_specificity_svm <- NULL

vect_accuracy_knn <- NULL
vect_sensitivity_knn <- NULL
vect_specificity_knn <- NULL

pt<-proc.time()
for(i in 1:n){
  
  ############ Initialisation paramètres ############
  
  napp=1000
  ch1=4
  ch2=9
  
  ########### Echantillons d'apprentissage et de test ##############
  
  train.1SVM=train[sample(which(train$y==ch1 | train$y==ch2) ,napp),]
  test.1SVM=test[which(test$y==ch1 | test$y==ch2 ),]
  test.1SVM$y = as.factor(ifelse(test.1SVM$y == 4,0,1))
  train.1SVM$y = as.factor(ifelse(train.1SVM$y == 4 , 0, 1))
  
  
  ##### enlever les colonnes nulles ########
  #ind=which(colMeans(train.1SVM)==0)
  #train.1SVM=train.1SVM[,-ind]
  #test.1SVM=test.1SVM[,-ind]
  
  ###### Entrainement du modèle svm ##########
  
  model_binary_svm =ksvm(train.1SVM$y ~ .,data = train.1SVM, kernel="rbfdot", C=1, kpar="automatic",scaled=F)
  
  ######## Prédiction svm ############
  
  eval_binary_svm = predict(model_binary_svm,newdata = test.1SVM)
  
  
  vect_accuracy_svm <- c(vect_accuracy_svm,confusionMatrix(eval_binary_svm,test.1SVM$y)$overall[1])
  vect_sensitivity_svm <- c(vect_sensitivity_svm,confusionMatrix(eval_binary_svm,test.1SVM$y)$byClass[1])
  vect_specificity_svm <- c(vect_specificity_svm,confusionMatrix(eval_binary_svm,test.1SVM$y)$byClass[2])
  
  ###### Entrainement du modèle knn ##########
  Accuracy_knn = 0
  Sensitivity_knn = 0
  Specificity_knn = 0
  
  # Optimisation du k 
  
  for(j in 1:15){
    model_binary_knn <- knn(train = train.1SVM[,-785], test = test.1SVM[,-785],cl = train.1SVM$y, k = j)
    
    
    cm_knn <- confusionMatrix(model_binary_knn, test.1SVM$y)
    
    
    if(cm_knn$overall[['Accuracy']] > Accuracy_knn){
      Accuracy_knn = cm_knn$overall[['Accuracy']]
      Sensitivity_knn = cm_knn$byClass[1]
      Specificity_knn = cm_knn$byClass[2]
      best_k = j
    }
  }
  cat("Best accuracy model reached for k = ",best_k,"\n", "with accuracy = ",Accuracy_knn,sep=" ","\n")
  
  vect_accuracy_knn <- c(vect_accuracy_knn,Accuracy_knn)
  vect_sensitivity_knn <- c(vect_sensitivity_knn,Sensitivity_knn)
  vect_specificity_knn <- c(vect_specificity_knn,Specificity_knn)
  
}

proc.time()-pt

mean_accuracy_svm = mean(vect_accuracy_svm)
mean_sensitivity_svm = mean(vect_sensitivity_svm)
mean_specificity_svm = mean(vect_specificity_svm)

mean_accuracy_knn = mean(vect_accuracy_knn)
mean_sensitivity_knn = mean(vect_sensitivity_knn)
mean_specificity_knn = mean(vect_specificity_knn)



table_means_svm = data.frame(mean_accuracy_svm,mean_sensitivity_svm,mean_specificity_svm)
table_means_knn = data.frame(mean_accuracy_knn,mean_sensitivity_knn,mean_specificity_knn)

setwd("C:/Users/33650/Documents/CoursL2/MachineLearning")
write.table(table_means_svm,"Means_binary_svm.csv",row.names=FALSE,dec=".", na=" ",append=FALSE)
write.table(table_means_knn,"Means_binary_knn.csv",row.names=FALSE,dec=".", na=" ",append=FALSE)
