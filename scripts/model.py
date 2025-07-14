from sklearn.ensemble import RandomForestClassifier

def build_and_train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    return model