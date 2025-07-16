import h5py 

from model import LogRegression

class ImageClassifier:
    def __init__(self):
        self.x_train = None 
        self.y_train = None 
        self.x_test = None 
        self.y_test = None
        self.model = None 

    def load_dataset(self, train_set_path: str = "dataset//train_catvnoncat.h5", test_set_path: str = "dataset//test_catvnoncat.h5"):
        train_set_file = h5py.File(train_set_path, 'r')
        test_set_file = h5py.File(test_set_path, 'r')

        x_train = train_set_file['train_set_x'][:]
        y_train = train_set_file['train_set_y'][:]
        x_test = test_set_file['test_set_x'][:]
        y_test = test_set_file['test_set_y'][:]

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

    def train(self):
        self.model = LogRegression(self.x_train.shape[0], 2000, 0.001)
        self.model.fit(self.x_train, self.y_train)
    
    def test(self):
        if self.model is None:
            print("Model not trained yet. Please train first.")
            return
        self.model.test(self.x_test, self.y_test)


if __name__ == "__main__":
    classifier = ImageClassifier()
    classifier.load_dataset()
    classifier.train()
    classifier.test()
    