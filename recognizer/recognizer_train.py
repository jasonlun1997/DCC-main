import os
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from ignite.handlers import Checkpoint, DiskSaver
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, ConfusionMatrix

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
batch_size = 64
scale_factor = 2

# load training and validation dataset
data_dir = './dataset/DL/dataset_2x_training-position'  # position #scheme
def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.CenterCrop(int(432/scale_factor)),
                                    transforms.ToTensor()])
    all_images = datasets.ImageFolder(data_dir, transform=transform)
    # 80% images for training
    train_images_len = int(len(all_images) * 0.8)
    # 20% images for validating
    val_images_len = int((len(all_images) - train_images_len))
    train_data, val_data = random_split(
        all_images, [train_images_len, val_images_len])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return (train_loader, val_loader), all_images.classes
(train_loader, val_loader), classes = get_data_loaders(data_dir, batch_size)
print(classes)

# model param
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = models.swin_transformer.swin_b(weights='DEFAULT')
print("model.head.in_features: ", model.head.in_features)
for param in model.parameters():
    param.requires_grad = False
n_inputs = model.head.in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.head = last_layer
print(model.head.out_features)
model = model.to(device)

# training config
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=0)
training_history = {'accuracy': [], 'loss': []}
validation_history = {'accuracy': [], 'loss': []}
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model,
                                        device=device,
                                        metrics={
                                            'accuracy': Accuracy(),
                                            'loss': Loss(criterion),
                                            'cm': ConfusionMatrix(len(classes))
                                        })


# create a event handler to show our training progress
@trainer.on(Events.ITERATION_COMPLETED)
def log_a_dot(engine):
    print(".", end="")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    training_history['accuracy'].append(accuracy)
    training_history['loss'].append(loss)
    print()
    print(
        f"Training results - Epoch:{trainer.state.epoch} Avg accuracy: {accuracy} Loss: {loss}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    validation_history['accuracy'].append(accuracy)
    validation_history['loss'].append(loss)
    print()
    print(
        f"Validation results - Epoch:{trainer.state.epoch} Avg accuracy: {accuracy} Loss: {loss}")


# model checkpoint
to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
checkpoint_dir = "./pth/2x/DL/"  # position #scheme
checkpoint = Checkpoint(
    to_save,
    DiskSaver(checkpoint_dir, require_empty=False),
    n_saved=None,
    global_step_transform=global_step_from_engine(trainer)
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint)

# start training
trainer.run(train_loader, max_epochs=150)

# show metrics
fig, axs = plt.subplots(2, 2)
fig.set_figheight(6)
fig.set_figheight(14)
axs[0, 0].plot(training_history["accuracy"])
axs[0, 0].set_title("Training Accuracy")

axs[0, 1].plot(validation_history["accuracy"])
axs[0, 1].set_title("Validation Accuracy")

axs[1, 0].plot(training_history["loss"])
axs[1, 0].set_title("Training Loss")

axs[1, 1].plot(validation_history["loss"])
axs[1, 1].set_title("Validation Loss")
plt.show()
