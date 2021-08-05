from PytorchModel import PytorchModel
import time
import copy

class ModelTrainer:

    ###TODO integrate data into here correctly
    def __init__(self, model: PytorchModel, data):
        self.model = model
        self.data = data

    def training_set_eval(self):
        self.model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader[phase]:
            print(labels)
            inputs = inputs.permute([0, 3, 2, 1])
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero out the gradients before training
            optimizer.zero_grad()

            # set gradient calculations ON for training
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(dataloader[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)



    def training_steps(self, dataloader, model, criterion, optimizer, scheduler, epochs=25, **inputs):
        start = time.time()

        val_acc_history = []
        val_loss_history = []
        lr_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            print(f'Epoch {epoch +1}/{epochs}')
            print('- ' * 15)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloader[phase]:
                    print(labels)
                    inputs = inputs.permute([0, 3, 2, 1])
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero out the gradients before training
                    optimizer.zero_grad()

                    # set gradient calculations ON for training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloader[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

                print(f'{phase} loss: {epoch_loss:.3f}, accuracy: {epoch_acc:.3f}')

                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                if phase == 'valid':
                    val_acc_history.append(epoch_acc.item())
                    val_loss_history.append(epoch_loss)
                    lr_history.append(optimizer.param_groups[0]['lr'])


            print()

        time_elapsed = time.time() - start
        print(f'Training time: {time_elapsed // 60}m {time_elapsed % 60}s')
        print()

        model.load_state_dict(best_model_wts)

        return model, val_acc_history, val_loss_history, lr_history