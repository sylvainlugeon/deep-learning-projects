import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from dlc_practical_prologue import *
import time
          
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def train_model(model, rounds, nb_epochs=100, batch_size=100, validation=True, verbose=False):
                
    nb_epochs_shown = 10
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    
    # tensors for averaging over the rounds
    times = torch.zeros(rounds)
    train_acc = torch.zeros(rounds)
    test_acc = torch.zeros(rounds)
    train_loss = torch.zeros(rounds, nb_epochs + 1)
    validation_loss = torch.zeros(rounds, nb_epochs + 1)
    
    for r in range(rounds):
        
        # new data
        train_input, train_target, train_classes, val_input, val_target, val_classes = generate_pair_sets(nb=1000)
        
        # reset weights
        model.apply(weight_reset) 
        
        t0 = time.time()

        for e in range(nb_epochs):
            
            # store the train and validation loss for each epoch and round 
            if validation:
                model.train(False) # deactivate dropout
                train_loss[r,e] = criterion(model(train_input)[0], train_target)
                validation_loss[r,e] = criterion(model(val_input)[0], val_target)
                model.train(True) # activate dropout
            
            # updating the model
            for input, targets, classes in zip(train_input.split(batch_size), 
                                               train_target.split(batch_size), 
                                               train_classes.split(batch_size)):
                
                output = model(input)
                loss = criterion(output[0], targets)
                if output[1] is not None:
                    c_aux = 0
                    c_final = 1
                    loss_aux1 = criterion(output[1][:, :10], classes[:,0])
                    loss_aux2 = criterion(output[1][:, 10:], classes[:,1])
                    loss = c_final * loss + c_aux * (loss_aux1 + loss_aux2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print the loss for a given number of epochs - used for direct feedback
            if verbose:
                if((e + 1) % int(nb_epochs / nb_epochs_shown) == 0):
                    print("Epoch {} | Train loss : {}".format(e+1, loss))
                    
        # final loss            
        if validation:
            model.train(False) # deactivate dropout
            train_loss[r,nb_epochs] = criterion(model(train_input)[0], train_target)
            validation_loss[r,nb_epochs] = criterion(model(val_input)[0], val_target)
            model.train(True) # activate dropout

        t1 = time.time()
        
        times[r] = t1-t0
        train_acc[r] = accuracy(model, train_input, train_target)
        test_acc[r] = accuracy(model, val_input, val_target)
        
        print('Round {} done.'.format(r))
        
    print('--------------')
    
    total_trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model : {} \n".format(model.__class__.__name__ ) + \
          "Number of trained parameters : {} \n".format(total_trained_params) + \
          "Size of mini-batches : {}\n".format(batch_size) + \
          "Averaged on {} rounds \n".format(rounds+1) + \
          "    Time for {} epochs : {:.2f}s\n".format(nb_epochs, times.mean().item()) + \
          "    Train accuracy : {:.3f} \n".format(train_acc.mean().item()) + \
          "    Test accuracy : {:.3f}".format(test_acc.mean().item()))
    
    return train_loss.detach().mean(dim=0), validation_loss.detach().mean(dim=0)

def accuracy(model, test_input, test_target):
    nb_samples = test_input.shape[0]
    model.train(False) # deactivate dropout
    output = model(test_input)[0]
    model.train(True) # activate dropout
    output_int = torch.zeros(nb_samples)
    for i in range(nb_samples):
        if output[i][0] <= output[i][1]: # first digit lesser or equal
            output_int[i] = 1
    nb_errors = (output_int - test_target).type(torch.BoolTensor).sum().item()
    return (nb_samples - nb_errors) / nb_samples