import pandas as pd
import numpy as np
from PIL import Image
import io

from model import DeepNeuralNetwork

class MultiImageClassifier:  # Fixed typo from MultiImageClassier
    def __init__(self):
        self.x_train = None 
        self.y_train = None 
        self.x_test = None 
        self.y_test = None
        self.model = None

    def _extract_image(self, image_dict):
            png_bytes = image_dict['bytes']
            image = Image.open(io.BytesIO(png_bytes)).convert('L')
            return np.asarray(image).flatten()

    def load_dataset(self, train_set_path: str = "datasets//train-mnist.parquet", test_set_path: str = "datasets//test-mnist.parquet"):
        # Load and process training data
        train_df = pd.read_parquet(train_set_path)
        self.x_train = np.stack([self._extract_image(img) for img in train_df['image']]).astype('float32') / 255.0
        self.x_train = self.x_train.T
        labels = train_df['label'].values
        num_classes = labels.max() + 1
        self.y_train = np.eye(num_classes)[labels].T

        # Load and process test data
        test_df = pd.read_parquet(test_set_path)
        self.x_test = np.stack([self._extract_image(img) for img in test_df['image']]).astype('float32') / 255.0
        self.x_test = self.x_test.T
        test_labels = test_df['label'].values
        self.y_test = np.eye(num_classes)[test_labels].T

    def train(self):
        if self.x_train is None or self.y_train is None:
            print("Data not loaded. Please load dataset first.")
            return
            
        input_size = self.x_train.shape[0]  # 784 is a common size for MNIST
        output_size = self.y_train.shape[0]  # 10 is the number of classes for each digit
        
        print(f"Input size: {input_size}, Output size: {output_size}")
        
        self.model = DeepNeuralNetwork(
            dimensions=[input_size, 128, 64, output_size], 
            activations=['relu', 'relu', 'softmax'],  # Changed to softmax
            iterations=1000,  
            learning_rate=0.01
        )
        
        self.model.fit(self.x_train, self.y_train)
    
    def test(self):
        if self.model is None:
            print("Model not trained yet. Please train first.")
            return
        
        accuracy = self.model.test(self.x_test, self.y_test)
        return accuracy


if __name__ == "__main__":
    classifier = MultiImageClassifier()
    classifier.load_dataset()
    classifier.train()
    classifier.test()