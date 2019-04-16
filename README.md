# Bank-Marketing
The model is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y). 

Deep Neural Network has been used to train the classifier.
The Neural Network consists os 4 layers of the following configuration:-
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
	The preprocessing phase mainly includes importing the data set. 
	Then separating features and labels(features: 1-20, labels: 21). 
	This is followed by Encoding of the dataset. I have used ColumnTransformer from sklearn.compose.
	After that we split training and test set. 20% randomly selected rows are used for testing.
This completes our data preprocessing phase

### 2. Classification
	As mention earlier, I am going to use Neural Network for classification since the dataset is huge.
	The Neural Network is first initialised(65 Nodes) (Activation function: Rectifier Function)
	The Input layer and the second hidden layer is added (35 Nodes)(Activation function: Rectifier Function)
	The Output layer is added(1 Node)(Activation function: Sigmoid)
	Classifier is trained using train dataset
Classifier is tested using test dataset. 
Confusion Matrix is created to evaluate the accuracy of the classifier. 
Accuracy obtained: 91.91%
	

