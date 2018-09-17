# Handwritten-Devanagari-Character-Recognition-using-Neural-Networks
Handwritten devanagari character (UCI Dataset) recognition has been performed using Neural Networks. Features of the characters are extracted using Convlution Neural Network and Deep Neural Networks. The extracted features are then used to predict the characters using Classifiers: Random Forest, KNN and Multi-layer perceptron. Efficiencies of each are studied under different scenarios.


Sample.rar contains the UCI dataset for characters in folder Train_Data_Full and a limited characters in folder Train_Partial_Data.
It also contains cropped images used for prediction in the Predict_Cropped_Images folder which are predicted by the model after being trained on full character images.

To extract individual characters from a word in the image, Google’s Tesseract OCR was used.

Data preprocessing:
The dataset picked up from UCI contains training and test data. Each having 36 Devanagari characters. For each character a folder is created containing the name of the character in English. Each folder contains 1700 images of the respective character. The target labels (the character name in English) is not given separately.
Thus, data is preprocessed by extracting the character name from the folder and storing into and label array which is further used for training the model.
Each image is a 32 * 32 grayscale image which is to be converted into array and then flattened a stored in an image matrix to train the model.

Feature Extraction:
Convolution Neural networks (CNN) has been the best feature extraction Neural network used so far by various authors. Here, the scope has been tested and experimented by using Dense Neural networks in combination to CNN. “RELU” activation function is used for input and hidden layers and “sigmoid” activation function is used in the output layer.The features extracted from the dense layers are then passed to the classification model.

Following classifiers which are popular for multiclass classification are used to classify the target labels from the extracted features:
1. Random Forest Classifier:

● A random forest fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy.

● For model trained on all 36 characters the accuracy of Random Forest Classifier was found to be in a range of 68% to 78%

● For a model trained and validated on limited number of characters the accuracy was around 85% to 90%

● A model where the cropped images were used for training and validation the CNN-Random Forest accuracy dropped to 22% to 30%

2. Multi-Layer Perceptron Classifier:

● One of the popular classifiers for multiclass classification which uses stochastic gradient descent to optimize log-loss function.

● For fully trained and validated network across 36-character images the accuracy of MLP was around 74% to 80 %

● For model validated on limited number of character images the accuracy of this classifier was between 87% to 92%

● In experiment to train and verify the model on cropped images, MLP gave an accuracy of around 55%

3. K-Nearest Neighbor Classifier:

● K Nearest Neighbor classifier implements the k-nearest neighbors vote.

● For all 36 characters trained and validated model, KNN had a classification accuracy of around 78% to 82%

● KNN had an accuracy ranging between 88% to 92% for model trained and validated on small number of target classes.

● The accuracy of KNN dropped down to 25% when the model was trained and validated on the cropped images




EXPERIMENTS

Various experiments were performed to analyze the effect of extracting features from Neural Networks and then using Classifiers to predict the target using those features. Each of the experiment performed has been explained in detailed below:

1) Extract features using Neural Networks and predict target using Classifiers

● All the currently available work on character recognition of Devanagari script has been done by implementing CNN and DNN only.

● An attempt was made to modify this approach by integrating the Neural Networks and multi class classifiers.

● In this experiment, Neural Network was configured to extract the features from the images.

● Read images were then fed to this network consisting of multiple convolution, pooling and dense layers.

● The output of the second last layer (dense layer) from the Neural Networks consisting of the features was extracted after passing the inputs to the model.

● This output was then divided as train and test data to evaluate classifier performance.

● The classifiers were trained using these extracted features and their corresponding labels from the train data.

● Test data created above was used to assess the performance of the classifiers.

● In order to better understand the variations, different classifiers like Random Forest, Multi-Layer Perceptron and K-Nearest Neighbors were evaluated.

● Performance of each classifier showed variations depending on the type of data used to train and evaluate the model.

KEY FINDINGS:

● Small variations were observed in the accuracies of different classifiers which took the features extracted from CNN as input and predicted the target labels on test data.

● The accuracy of the CNN model, if used for classification, was found out to be in a range of 70% to 80% for 10 epochs.

● If the classifiers were fed with features from this neural network, then their classification accuracy was in range of 72% to 81%.

● Thus, it could be seen that using the classifiers led to a small increase in the accuracy.

2) Train the model on cropped character images and predict full character images

● In real life scenario it may be possible that the available images may contain cropped/partial characters due to poor data quality.

● An experiment was performed to evaluate the effect of training the model using the cropped images and then using it to predict full character images.

● Due to unavailability of data existing data with cropped images and time constraints, only a small amount of data was used for training and validating the model.

● This however displayed interesting results and can be made a part of future scope.

● In this experiment, a few images of each character were cropped manually and given to the model for training and validation.

● Full character images were then passed to the model and model was then used to predict those images.

● The results showed that model could identify the pattern from cropped images and could be used predict the full images.

● This was an important finding as it proved that the features extracted from neural networks could be used by the classifiers to correctly predict images.

KEY FINDINGS:

● This experiment yielded some interesting results and hypothesis.

● It was observed that when the images containing cropped characters were used to train the model, the model was able to learn the features of images.

● Here the CNN model was passed with cropped images and the features extracted were used by classifiers to predict labels.

● Since only cropped images were used for training and validation the accuracy of the model was less (around 25% to 55%) as expected only a few cropped parts could be predicted by the model.

● However, it was observed that when full character images were passed to such a trained model for prediction, at least 2 of the 3 classifiers correctly predicted the labels.

● It was because the classifiers had an intimation of the features from the cropped images which they could locate in the full images and correctly identify them leading to better accuracy.

3) Training the model on full character images and predict the cropped images

● Following the reverse methodology implemented above, an experiment was performed to train the model on full character images and then predict an image with cropped character in it.

● The primary motivation of this experiment was to analyze how the model performed on receiving a partial character for prediction.

● In order to perform this experiment, the model was first trained and evaluated on all 36 target character labels.

● Certain cropped images were then passed to the model for prediction to analyze if it correctly predicted the images.

● Similar to above experiment, the model showcased some interesting findings and we could hypothesize some of the reasons for such behavior.

● Based on this experiment it could be said that the unlike the experiment above, the model would not be efficient to predict cropped images after training it on full character images.

KEY FINDINGS:

● Unlike the above experiment, this experiment provided with some unusual results.

● Here the model was trained on full character images and then used to predict the cropped images.

● Such an arrangement led to the mediocre performance of the model due to the ambiguity introduced by cropping of the images.

● The model yielded good prediction accuracy when trained and validated on the full character image data.

● But when cropped images were provided, the model led to misclassification possibly since cropping of the image may lead for the model to interpret it as a different character.

● Since Devanagari contains many such similar looking characters, a small modification can lead to a complete change of character.

● For e.g. if character ‘ka’ is cut into 2 exact vertical halves then the left image is identical to another character ‘waw’.

● This lead to mediocre performance by the model in such an arrangement.

4) Predict a printed Devanagari character by a model trained to classify handwritten characters

● To bring some novelty into the existing work, a printed Devanagari character was passed to the model for prediction which was trained on the handwritten characters.

● To pass the printed characters, first images of words with Devanagari characters were read and individual characters from those words were extracted.

● To extract individual characters from a word in the image, Google’s Tesseract OCR was used.

● After extracting the character, each character was stored in a string as Unicode character.

● Each of these Unicode characters were then rendered and stored as image files which could be later passed to the model.

● The images created above were read and converted into appropriate format like gray-scale image, 32X32 size, etc.

● The images were then passed to the model trained on handwritten character images and predicted using the classifiers which took features from the model.

● After analyzing the performance, the model was re-trained using limited number of character targets and then printed image passed for prediction.

KEY FINDINGS:

● As a preliminary experiment, its results largely depended on the characters passed for prediction and the classifiers used to predict the target label.

● The results were also impacted by the data used for training at that instant and also the uniqueness of the character to be predicted.

● Characters like ‘ka’ and ‘ma’ which are mostly unique in the script could be predicted correctly by the classifiers.

● Whereas the characters like ‘ha’ or ‘yaw’ which have similar looking identities may lead to misclassification.


CONCLUSION: 
Handwritten character recognition is still a research area of burning pattern recognition. Character recognition of handwritten Devanagari script is a difficult task considering the similarities between its characters. With the use of Neural Networks for extracting the important features of the character in the images has been very useful in mining the characteristics of the image and hence making classification of the characters simple using various multiclass classifiers. Moreover, experimenting with full and partial images of the characters using different neural network architecture has helped understand how the quality of the extracted features change, thus affecting the classification models and its accuracy. To conclude, Handwritten character recognition, Image processing, Feature extraction, neural networks are the various popular fields of research and the insights of these topics can be obtained from the report.
