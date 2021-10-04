##### Package installation to read images #####

library(BiocManager)
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("EBImage")

library(EBImage)

##### Directory path #####

image_dir <- "C:/Users/33650/Documents/CoursL2/MachineLearning/dogs-vs-cats/train/train"

##### Images examples #####

example_cat_image <- readImage(file.path(image_dir, "cat.0.jpg"))
display(example_cat_image)

example_dog_image <- readImage(file.path(image_dir, "dog.0.jpg"))
display(example_dog_image)

##### Images pre-processing #####

width <- 28
height <- 28
## pbapply is a library to add progress bar *apply functions
## pblapply will replace lapply
install.packages("pbapply")
library(pbapply)
extract_feature <- function(dir_path, width, height, is_cat = TRUE, add_label = TRUE) {
  img_size <- width*height
  ## List images in path
  images_names <- list.files(dir_path)
  if (add_label) {
    ## Select only cats or dogs images
    images_names <- images_names[grepl(ifelse(is_cat, "cat", "dog"), images_names)]
    ## Set label, cat = 0, dog = 1
    label <- ifelse(is_cat, 0, 1)
  }
  print(paste("Start processing", length(images_names), "images"))
  ## This function will resize an image, turn it into greyscale
  feature_list <- pblapply(images_names, function(imgname) {
    ## Read image
    img <- readImage(file.path(dir_path, imgname))
    ## Resize image
    img_resized <- resize(img, w = width, h = height)
    ## Set to grayscale
    grayimg <- channel(img_resized, "gray")
    ## Get the image as a matrix
    img_matrix <- grayimg@.Data
    ## Coerce to a vector
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  ## bind the list of vector into matrix
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  ## Set names
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  if (add_label) {
    ## Add label
    feature_matrix <- cbind(label = label, feature_matrix)
  }
  return(feature_matrix)
}


##### data processed #####

cats_data <- extract_feature(dir_path = image_dir, width = width, height = height)
dogs_data <- extract_feature(dir_path = image_dir, width = width, height = height, is_cat = FALSE)
dim(cats_data)
dim(dogs_data)



cats_data$label
dogs_data$label

saveRDS(cats_data, "cat.rds")
saveRDS(dogs_data, "dog.rds")



library(class)
library(caret)
library(kernlab)
library(e1071)
set.seed(100)

##### Initialisation #####

n = 50

vect_accuracy_svm <- NULL
vect_sensitivity_svm <- NULL
vect_specificity_svm <- NULL

vect_accuracy_knn <- NULL
vect_sensitivity_knn <- NULL
vect_specificity_knn <- NULL


pt<-proc.time()
for(i in 1:n) {

##### train sample #####
  
indice <- sample(1:12500, 6000)
train_data <- rbind(cats_data[indice[1:1000],], dogs_data[indice[1:1000],])
validation_data <- rbind(cats_data[indice[2001:2250],], dogs_data[indice[2001:2250],])
test_data <- rbind(cats_data[indice[3001:5000],], dogs_data[indice[3001:5000],]) 


train_data$label <- as.factor(train_data$label)
test_data$label <- as.factor(test_data$label)
validation_data$label <- as.factor(validation_data$label)

##### model svm #####

model_svm <- ksvm(train_data$label ~. , data = train_data, kernel = "rbfdot", C = 1, 
                  kpar = "automatic", type = "C-svc")


eval_svm <- predict(model_svm, newdata = test_data)

cm_svm <- confusionMatrix(eval_svm, test_data$label)

vect_accuracy_svm <- c(vect_accuracy_svm,cm_svm$overall[1])
vect_sensitivity_svm <- c(vect_sensitivity_svm,cm_svm$byClass[1])
vect_specificity_svm <- c(vect_specificity_svm,cm_svm$byClass[2])


##### model knn #####

Accuracy_knn = 0
Sensitivity_knn = 0
Specificity_knn = 0

for(j in 1:15){

model_knn <- knn(train = train_data[,-1], test = validation_data[,-1], cl = train_data$label, k = j)

cm_knn <- confusionMatrix(model_knn, validation_data$label)


if(cm_knn$overall[['Accuracy']] > Accuracy_knn){
  Accuracy_knn = cm_knn$overall[['Accuracy']]
  Sensitivity_knn = cm_knn$byClass[1]
  Specificity_knn = cm_knn$byClass[2]
  best_k = j
}
}
cat("Best accuracy model reached for k = ",best_k,"\n", "with accuracy = ",Accuracy_knn,sep=" ","\n")

model_knn <- knn(train = train_data[,-1], test = test_data[,-1], cl = train_data$label, k = best_k)
cm_knn <- confusionMatrix(model_knn, test_data$label)

Accuracy_knn = cm_knn$overall[['Accuracy']]
Sensitivity_knn = cm_knn$byClass[1]
Specificity_knn = cm_knn$byClass[2]

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

