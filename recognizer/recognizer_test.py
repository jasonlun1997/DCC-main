import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from ignite.engine import create_supervised_trainer
from ignite.handlers import Checkpoint

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
batch_size = 64
scale_factor = 2

# load test dataset
data_dir = './dataset/dataset_2x_test-position'  # scheme #position
def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.CenterCrop(int(432/scale_factor)),
                                    transforms.ToTensor()])  
    all_images = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(all_images, batch_size=batch_size)
    return (test_loader), all_images.classes
(test_loader), classes = get_data_loaders(data_dir, batch_size)

# model param
model = models.swin_transformer.swin_b(weights='DEFAULT')
testing_history = {'accuracy': [], 'loss': []}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=0)
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
n_inputs = model.head.in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.head = last_layer
print(model.head.out_features)

# load model checkpoint 
for i in range(150): #range depent your training
    num = i+1
    print("epoch:"+str(num))
    # checkpoint_dir = "./pth/2x/position_swinB_epoch150/"  # position #scheme
    checkpoint_dir = "D:\SY_checkpoint_0324\position_swinB_epoch150/"
    checkpoint_fp = checkpoint_dir + "checkpoint_" + str(num) + ".pt"
    checkpoint = torch.load(checkpoint_fp, map_location=device)
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    test_loss = 0.0
    class_correct = np.zeros((len(classes)))
    class_total = np.zeros((len(classes)))
    model = model.to(device)
    model.eval()
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available(
        ) else np.squeeze(correct_tensor.cpu().numpy())
        # if data in GPU pass to CPU tensor and numpy array
        if len(target) == 64:
            for i in range(64):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    test_loss /= len(test_loader.dataset)
    testing_history['loss'].append(test_loss)
    print("Test Loss: {:.6f}".format(test_loss))

    for i in range(len(classes)):
        if class_total[i] > 0:
            print("Test Accuracy of {}: {} ({}/{})".format(
                classes[i], 100*class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])
            ))
        else:
            print(
                "Test Accuracy of {}: N/A (since there are no examples)".format(classes[i]))
    print("Test Accuracy Overall: {} ({}/{})\n".format(
        100*np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)
    ))
    testing_history['accuracy'].append(
        100*np.sum(class_correct) / np.sum(class_total))

#show metrics
fig, axs = plt.subplots(2, 1)
fig.set_figheight(6)
fig.set_figheight(14)

axs[0].plot(testing_history["accuracy"])
axs[0].set_title("Testing Accuracy")

axs[1].plot(testing_history["loss"])
axs[1].set_title("Testing Loss")

plt.show()
