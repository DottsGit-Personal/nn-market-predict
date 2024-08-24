import argparse
from data_loader import load_data
from model import create_model
from train import train_model
from evaluate import evaluate_model

def main(file_path, target_column, hidden_sizes):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_data(file_path, target_column)
    
    # Create model
    input_size = X_train.shape[1]
    output_size = 1  # Assuming regression task
    model = create_model(input_size, hidden_sizes, output_size)
    
    # Train model
    trained_model = train_model(model, X_train, y_train)
    
    # Evaluate model
    evaluate_model(trained_model, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network model.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing the dataset")
    parser.add_argument("target_column", type=str, help="Name of the column to predict")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 32], help="Sizes of hidden layers")
    
    args = parser.parse_args()
    
    main(args.file_path, args.target_column, args.hidden_sizes)