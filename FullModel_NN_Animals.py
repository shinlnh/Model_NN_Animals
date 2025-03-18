from Dataset_DUAUO_Animals import Animal_Dataset
from NN_Model_Animals import SimpleNeuralNetwork
from torchvision.transforms import ToTensor,Resize
from torchvision.transforms.v2 import Compose, ToPILImage
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import classification_report


if __name__== '__main__':
    num_epochs = 100
    transform = Compose([
        Resize((32,32)),
        ToTensor()
    ])
    training_dataset = Animal_Dataset(root="DUAUO_ANIMALS", train=True,transform=transform)
    training_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    drop_last=True
    )
    testing_dataset = Animal_Dataset(root="DUAUO_ANIMALS", train=False,transform=transform)
    testing_dataloader = DataLoader(
    dataset=testing_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=False,
    drop_last=False
    )
    model = SimpleNeuralNetwork(num_class=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr= 1e-3,momentum=0.9)
    num_iters = len(training_dataloader)
    for epoch in range(num_epochs):
        model.train()
        for iter,(images,labels) in enumerate(training_dataloader):
            #forwards
            prediction_train = model(images)
            loss_value = criterion(prediction_train,labels)
            # print("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch+1,num_epochs,iter+1,num_iters,loss_value))

            #backwards

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # Evaluation after each epoch
        model.eval()
        all_prediction = []
        all_labels = []
        for iter, (images, labels) in enumerate(testing_dataloader):
            all_labels.extend(labels)
            with torch.no_grad():
                prediction_test = model(images)
                indices = torch.argmax(prediction_test.cpu(), dim=1)
                all_prediction.extend(indices)
                loss_value = criterion(prediction_test, labels)

        # Convert labels and predictions to scalar values
        all_labels = [label.item() for label in all_labels]
        all_prediction = [prediction.item() for prediction in all_prediction]

        # Print after each epoch
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print(classification_report(all_labels, all_prediction,zero_division=1))






