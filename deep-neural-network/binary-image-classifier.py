import h5py 

from model import DeepNeuralNetwork

class BinaryImageClassifier:
    def __init__(self):
        self.x_train = None 
        self.y_train = None 
        self.x_test = None 
        self.y_test = None
        self.model = None 

    def load_dataset(self, train_set_path: str = "datasets//train_catvnoncat.h5", test_set_path: str = "datasets//test_catvnoncat.h5"):
        try:
            train_set_file = h5py.File(train_set_path, 'r')
            test_set_file = h5py.File(test_set_path, 'r')
            
            x_train = train_set_file['train_set_x'][:]
            y_train = train_set_file['train_set_y'][:]
            x_test = test_set_file['test_set_x'][:]
            y_test = test_set_file['test_set_y'][:]
            
            # Close files to prevent resource leaks
            train_set_file.close()
            test_set_file.close()
            
        except FileNotFoundError as e:
            print(f"Dataset file not found: {e}")
            return
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # Reshape data, to make sure 64x64x3 is presented as a flat vector  
        x_train_reshaped = x_train.reshape(x_train.shape[0], -1).T
        x_test_reshaped = x_test.reshape(x_test.shape[0], -1).T
        y_train_reshaped = y_train.reshape(1, y_train.shape[0])
        y_test_reshaped = y_test.reshape(1, y_test.shape[0])
        
        # Normalize pixels
        x_train_reshaped = x_train_reshaped / 255.0
        x_test_reshaped = x_test_reshaped / 255.0

        self.x_train = x_train_reshaped
        self.x_test = x_test_reshaped
        self.y_train = y_train_reshaped
        self.y_test = y_test_reshaped

        print(f"Training data shape: {self.x_train.shape}")
        print(f"Test data shape: {self.x_test.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test labels shape: {self.y_test.shape}")

    def train(self):
        if self.x_train is None or self.y_train is None:
            print("Data not loaded. Please load dataset first.")
            return
            
        self.model = DeepNeuralNetwork([self.x_train.shape[0],10,10,1], ['leaky_relu', 'leaky_relu', 'sigmoid'], 2000, 0.001)
        self.model.fit(self.x_train, self.y_train)
    
    def test(self):
        if self.model is None:
            print("Model not trained yet. Please train first.")
            return
        self.model.test(self.x_test, self.y_test)


if __name__ == "__main__":
    classifier = BinaryImageClassifier()
    classifier.load_dataset()
    classifier.train()
    classifier.test()
