import torch
from torch import nn
from torch.nn import functional as F
import time
          
def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def accuracy(model, test_input, test_target):
    nb_samples = test_input.shape[0]
    model.train(False) # deactivate dropout
    output = model(test_input)
    model.train(True) # deactivate dropout
    output_int = torch.zeros(nb_samples)
    for i in range(nb_samples):
        if output[i][0] <= output[i][1]: # first digit lesser or equal
            output_int[i] = 1
    nb_errors = (output_int - test_target).type(torch.BoolTensor).sum().item()
    return (nb_samples - nb_errors) / nb_samples

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def train_model(model, train_input, train_target, val_input, val_target, rounds, validation=True, verbose=False):
    
    model.apply(weight_reset) # reset weights
    
    batch_size, nb_epochs = 100, 200
    
    nb_epochs_shown = 5
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    # tensors for averaging over the rounds
    times = torch.zeros(rounds)
    train_acc = torch.zeros(rounds)
    test_acc = torch.zeros(rounds)
    train_loss = torch.zeros(rounds, nb_epochs + 1)
    validation_loss = torch.zeros(rounds, nb_epochs + 1)
    
    for r in range(rounds):
            
        t0 = time.time()

        for e in range(nb_epochs):
            
            # store the train and validation loss for each epoch and round 
            if validation:
                model.train(False) # deactivate dropout
                train_loss[r,e] = criterion(model(train_input), train_target)
                validation_loss[r,e] = criterion(model(val_input), val_target)
                model.train(True) # deactivate dropout
            
            # updating the model
            for input, targets in zip(train_input.split(batch_size),train_target.split(batch_size)):
                output = model(input)
                loss = criterion(output, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print the loss for a given number of epochs - used for direct feedback
            if verbose:
                if((e + 1) % int(nb_epochs / nb_epochs_shown) == 0):
                    print("Epoch {} | Loss : {}".format(e+1, loss))
                    
        # final loss            
        if validation:
            model.train(False) # deactivate dropout
            train_loss[r,nb_epochs] = criterion(model(train_input), train_target)
            validation_loss[r,nb_epochs] = criterion(model(val_input), val_target)
            model.train(True) # deactivate dropout

        t1 = time.time()
        
        times[r] = t1-t0
        train_acc[r] = accuracy(model, train_input, train_target)
        test_acc[r] = accuracy(model, test_input, test_target)
        
        print('Round {} done.'.format(r))
        
    print('--------------')
    
    total_trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model : {} \n".format(model.__class__.__name__ ) + \
          "Number of trained parameters : {} \n".format(total_trained_params) + \
          "Size of mini-batches : {}\n".format(batch_size) + \
          "Averaged on {} rounds \n".format(rounds) + \
          "    Time for {} epochs : {:.2f}s\n".format(nb_epochs, times.mean().item()) + \
          "    Train accuracy : {:.3f} \n".format(train_acc.mean().item()) + \
          "    Test accuracy : {:.3f}".format(test_acc.mean().item()))
    
    return train_loss.detach().mean(dim=0), validation_loss.detach().mean(dim=0)