from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import graphviz


def visualize_classifier_name(clf, output_name):
	dot_data = tree.export_graphviz(clf, out_file=None) 
	graph = graphviz.Source(dot_data) 
	graph.render(f"1_decision_trees_and_random_forest/{output_name}") 


def train_classifiers_on_dataset(X_train, X_test, y_train, y_test):
	# Decision Tree classifier definitions
	clf = tree.DecisionTreeClassifier(criterion='gini')
	clf = clf.fit(X_train, y_train)
	visualize_classifier_name(clf, output_name='Decision Tree with gini')

	print("Evaluation report for gini based decision tree")
	print(classification_report(y_test, clf.predict(X_test)))

	clf = tree.DecisionTreeClassifier(criterion='entropy')
	clf = clf.fit(X_train, y_train)
	visualize_classifier_name(clf, output_name='Decision Tree with information gain')

	print("Evaluation report for entropy based decision tree")
	print(classification_report(y_test, clf.predict(X_test)))

	# Random forest with 100 estimators on gini criterion
	# clf = RandomForestClassifier(random_state=42,  n_estimators=100, criterion='gini')
	# clf.fit(X_train, y_train)
	#
	# print("Evaluation report for gini based RF")
	# print(classification_report(y_test, clf.predict(X_test)))


# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
train_classifiers_on_dataset(X_train, X_test, y_train, y_test)

X, y = make_classification(n_samples=5000, n_features=25,
                           n_informative=12, n_redundant=0,
                           random_state=0, shuffle=False,
                           n_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

train_classifiers_on_dataset(X_train, X_test, y_train, y_test)