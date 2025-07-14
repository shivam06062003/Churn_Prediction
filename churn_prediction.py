from scripts.data_preprocessing import preprocess_data
from scripts.model import build_and_train_model
from scripts.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    X, y = preprocess_data('data/telco_customer_churn.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_and_train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()