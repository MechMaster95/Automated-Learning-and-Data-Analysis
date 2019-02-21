import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import graphviz

data_path = "Data/hw2q5_numeric.csv"

data = np.genfromtxt(data_path, dtype=int, delimiter=',', names=True)
kf = KFold(n_splits=5)


k = 0
for train_index, test_index in kf.split(data):
	train_features = []
	train_classes = []
	test_features = []
	test_classes = []
	for i in train_index:
		subarray = []
		for j in range(1,5):
			subarray.append(data[i][j])
		train_features.append(subarray)
		train_classes.append([data[i][5]])
	for i in test_index:
		subarray = []
		for j in range(1,5):
			subarray.append(data[i][j])
		test_features.append(subarray)
		test_classes.append([data[i][5]])

	#Tree classifier
	tree_clf = tree.DecisionTreeClassifier(criterion="entropy")
	tree_clf = tree_clf.fit(train_features, np.array(train_classes))
	dot_data = tree.export_graphviz(tree_clf, out_file=None ,class_names=["No","Yes"], rounded=True) 
	graph = graphviz.Source(dot_data) 
	graph.render("q5_eyedata" + str(k))
	k+=1
	#Naive Bayes Classifier
	nb_clf = MultinomialNB()
	nb_clf.fit(train_features, np.array(train_classes))
	print("Split " + str(k-1) + " accuracy values:")
	print( "Tree:" + str(tree_clf.score(test_features, np.array(test_classes))))
	print( "NB:" + str(nb_clf.score(test_features, np.array(test_classes))) + '\n')





