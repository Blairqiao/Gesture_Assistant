import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    csv_file = "Data_Training/annotated_data.csv" 
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 1. Separate Features (X) and Labels (y)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    print(f"Total samples loaded: {len(df)}")
    print(f"Features per sample: {X.shape[1]}")
    
    # 2. Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Initialize the Random Forest Model
    # n_estimators=100 means it will build 100 decision trees
    # max_depth=15 prevents the model from memorizing the data (overfitting)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    
    # 4. Train the model
    model.fit(X_train, y_train)
    
    # 5. Evaluate the model on the unseen Test data
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Accuracy: {accuracy * 100:.2f}% ---")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the trained model to disk
    export_name = "Models/gesture_model.pkl"
    joblib.dump(model, export_name)
    print(f"\nSuccess! Model saved to {export_name}")

if __name__ == "__main__":
    main()