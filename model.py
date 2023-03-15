from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_ada_boost(X, y, n_estimators=50, max_depth=2, learning_rate=1):
    
    # Loading data and spliting it into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing a Decision Tree classifier as the weak learner
    dt = DecisionTreeClassifier(max_depth=max_depth)

    # Initializing the Adaboost classifier with n_estimators weak learners
    ada = AdaBoostClassifier(base_estimator=dt, n_estimators=n_estimators, learning_rate=learning_rate)

    # Performing cross-validation
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(ada, X_train, y_train, cv=10, scoring=scoring)

    # Training the Adaboost classifier on the training data
    ada.fit(X_train, y_train)

    # Making predictions on the testing data
    y_pred = ada.predict(X_test)

    # Evaluating the performance of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

