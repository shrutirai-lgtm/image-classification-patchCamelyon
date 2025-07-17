from data.data_loader import PatchCamelyonDataLoader
from model.ConvNet import ConvNet

def main():
    print("Loading PatchCamelyon dataset...")
    data_loader = PatchCamelyonDataLoader()
    
    # Get datasets
    train_data = data_loader.get_train_data()
    val_data = data_loader.get_val_data()
    test_data = data_loader.get_test_data()
    
    # Get dataset info
    image_shape = data_loader.get_image_shape()
    num_classes = data_loader.get_num_classes()
    print(f"Image shape: {image_shape}")
    print(f"Number of classes: {num_classes}")

    # Create model
    model = ConvNet(input_shape=image_shape, num_classes=num_classes)
    model.compile_model()
    model.train_model(train_data, val_data)
    model.evaluate_model(test_data)
    model.summary()

if __name__ == "__main__":
    main()