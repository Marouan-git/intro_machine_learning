
library(class)
library(kernlab)
library(caret)

############### choix nombre d'échantillonnages ################

n=50

set.seed(100)

################ Initialisation matrices et vecteurs ##################

vect_accuracy_svm <- NULL
vect_accuracy_knn <- NULL

matrix_sensitivities_svm = matrix(0, ncol=10, nrow=50,byrow=T)
matrix_sensitivities_knn = matrix(0, ncol=10, nrow=50,byrow=T)

accuracy_svm = 0



pt<-proc.time()
for(i in 1:n){
  
  ############# Echantillon ################
  
  sample_indices <- sample(1: nrow(train), 2000) # extracting subset of 2000 samples for modelling
  train_svm_knn <- train[sample_indices, ]
  
  
  ############ Partitions ############
  
  partition <- createDataPartition(train_svm_knn$y,times=1,p=0.70,list=F)
  train_partition <- train_svm_knn[partition,]
  test_partition <- train_svm_knn[-partition,]
  

  
 
  
  ############ Combinaisons des paramètres svm ############
  
  # p = ncol(train_partition)
  # grid_svm <- expand.grid(cost=1,sigma=c(1/p,2/p,1/(2*p)))
  
  
  ##################### Modèle svm #######################

  
    
    model_svm <- ksvm(train_partition$y ~ ., data = train_partition ,kernel = "rbfdot", C = 1, 
                      kpar = "automatic", scaled = F )
    
      
    
    
    
    # if(cm_svm$overall[['Accuracy']] > Accuracy_svm){
    # Accuracy_svm = cm_svm$overall[['Accuracy']]
    #  best_cost = grid_svm[i,1]
    #  best_gamma = grid_svm[i,2]
   # }
 
   # cat("Best accuracy model reached for cost = ",best_cost,"\n","gamma = ",best_gamma,"\n", "with accuracy = ",Accuracy_svm,sep=" ","\n")
    
   # best_model_svm <- svm(train_partition$y ~ ., data = train_partition,kernel="radial",  
                         # cost = best_cost, gamma = best_gamma, scale = FALSE)
    
    
    ################ modèle knn ###############
    
    Accuracy_knn = 0
    
    for(j in 1:15){
      model_knn <- knn(train = train_partition[,-785], test = test_partition[,-785],cl = train_partition$y, k = j)
      
      
      cm_knn <- confusionMatrix(model_knn, test_partition$y)
       
      
      if(cm_knn$overall[['Accuracy']] > Accuracy_knn){
        Accuracy_knn = cm_knn$overall[['Accuracy']]
        best_k = j
      }
    }
    cat("Best accuracy model reached for k = ",best_k,"\n", "with accuracy = ",Accuracy_knn,sep=" ","\n")
    
    
    
    ################## test svm sur les 10000 ###################
    
    eval_svm <- predict(model_svm, newdata = test)
    
    cm_svm <- confusionMatrix(eval_svm, test$y)
    
  
   
    ################ Récupération accuracy sensitivities svm ###################
    
    accuracy_svm <- cm_svm$overall[['Accuracy']]
    sensitivities_svm <- cm_svm$byClass[,"Sensitivity"]
    
 
      
    vect_accuracy_svm <- c(vect_accuracy_svm,accuracy_svm)
      
    
    
    matrix_sensitivities_svm[i,] <- sensitivities_svm
    
    
    
    
    ################## Test knn sur 10000 avec meilleure valeur de k ###################
    
    best_model_knn <-  knn(train = train_partition[,-785], test = test[,-785],cl = train_partition$y, k = best_k)
    cm_best_knn <- confusionMatrix(best_model_knn, test$y)
    
    ################# Récupération accuracy et sensitivities knn ###################
    
    accuracy_best_knn <- cm_best_knn$overall[['Accuracy']]
    sensitivities_knn <- cm_best_knn$byClass[,"Sensitivity"]
    
  
      
    vect_accuracy_knn <- c(vect_accuracy_knn,accuracy_best_knn)
      
    matrix_sensitivities_knn[i,] <- sensitivities_knn
    
}

proc.time()-pt # 6776 s

matrix_sensitivities_svm
matrix_sensitivities_knn



mean_accuracy_svm = mean(vect_accuracy_svm)
mean_sensitivities_svm = colMeans(matrix_sensitivities_svm)

mean_accuracy_knn = mean(vect_accuracy_knn)
mean_sensitivities_knn = colMeans(matrix_sensitivities_knn)

setwd("C:/Users/33650/Documents/CoursL2/MachineLearning")

table_means_sensitivities <- data.frame(mean_sensitivities_svm,mean_sensitivities_knn)
table_means_accuracies <- data.frame(mean_accuracy_svm,mean_accuracy_knn)

write.table(table_means_accuracies,"Mean_accuracies.csv",row.names=FALSE,dec=".", na=" ",append=FALSE)
write.table(table_means_sensitivities,"Mean_sensitivities.csv",row.names=FALSE,dec=".", na=" ",append=FALSE)

 

