from src.data_preprocessing import load_and_preprocess_data

(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
print(f'Train data shape: {x_train.shape}, Train labels shape: {y_train.shape}')
print(f'Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}')
