# Bank-Marketing
The model is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y). 

Deep Neural Network has been used to train the classifier.
The Neural Network consists of 4 layers of the following configuration:-
1. InputLayer         
2. HiddenLayer1          
3. HiddenLayer2         
4. OutputLayer

The accuracy obtained is: 91.95%

## Data Source:- 
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

## Approach:- 
The complete process can be divided into 2 phases:-
### 1. Data PreProcessing
	a. The preprocessing phase mainly includes importing the data set. 
	b. Then separating features and labels(features: 1-20, labels: 21). 
	c. This is followed by Encoding of the dataset. I have used ColumnTransformer from sklearn.compose.
	d. After that we split training and test set. 20% randomly selected rows are used for testing.
This completes our data preprocessing phase

### 2. Classification
	a. As mention earlier, I am going to use Neural Network for classification since the dataset is huge.
	b. The Neural Network is first initialised(65 Nodes) (Activation function: Rectifier Function)
	c. The Input layer and the second hidden layer is added (35 Nodes)(Activation function: Rectifier Function)
	d. The Output layer is added(1 Node)(Activation function: Sigmoid)
	e. Classifier is trained using train dataset
Classifier is tested using test dataset. 
Confusion Matrix is created to evaluate the accuracy of the classifier. 

Accuracy obtained: 91.91%

# Description about the content of the project

1. bank-additional-full.csv : The dataset used
2. classifier : The pickle file of the classifer to directly use the classifier without training it again
3. program.py : The complete model including data preprocessing, training and testing
4. tester.py : This uses the saved classifier to create confusion matrix on test dataset(testing)


