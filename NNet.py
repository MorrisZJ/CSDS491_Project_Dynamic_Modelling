import numpy as np
from eff_area_checker import is_neighbor, check_prediction
from mcl import get_mcl_cluster

def int_to_binary(n):
    binary_str = bin(n)[2:]  # Convert to binary string, removing '0b' prefix
    binary_arr = [int(digit) for digit in binary_str]  # Convert to binary array of integers
    return binary_arr


def pad_array(arr, size):
    if len(arr) >= size:
        return arr[:size]
    else:
        padding = [0] * (size - len(arr))
        return arr + padding


def data_preprocessing(data, data_size):
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    nnx_train = []
    nny_train = []
    nnx_test = []
    nny_test = []
    for i in range(len(x_train)):
        nnx_train.append(pad_array(x_train[i]['sound'] + [x_train[i]['start_state']], data_size))
        nny_train.append(y_train[i]['path'][-1])
        
    for i in range(len(x_test)):
        nnx_test.append(pad_array(x_test[i]['sound'] + [x_test[i]['start_state']], data_size))
        nny_test.append(y_test[i]['path'][-1])
        
    nnx_train = np.array(nnx_train)
    nny_train = np.array(nny_train)
    nnx_test = np.array(nnx_test)
    nny_test = np.array(nny_test)
    
    return nnx_train, nny_train, nnx_test, nny_test


def Multilayer_Net(data_size, 
                   board_size,
                   nnx_train, 
                   nny_train, 
                   nnx_test, 
                   nny_test, 
                   trans_mat,
                   output_size=25, 
                   num_epochs=150, 
                   batch_size=64, 
                   step_size=75,
                   lr=0.01):
    import torch
    import numpy as np
    import torch.nn as nn
    import copy
    
    cluster = get_mcl_cluster(trans_mat)
    
    # Define hyperparameters
    input_size = data_size
    learning_rate = lr
    # Convert data and labels to PyTorch tensors
    data = torch.from_numpy(nnx_train).float()
    labels = torch.from_numpy(nny_train).long()

    # Define neural network architecture
    model = torch.nn.Sequential(
        nn.Linear(input_size, 256),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.Linear(512, 1024),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 2048),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 2048),
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.Linear(1024,512),
        nn.Linear(512,256),
        nn.BatchNorm1d(256),
        nn.Linear(256, output_size),
        nn.MaxPool1d(1)
    )
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #weight_decay=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    currenr_best_acc = 0
    current_best_model = copy.deepcopy(model)

    def test(currenr_best):
        # Test the model
        with torch.no_grad():
            flag = False
            test_data = nnx_test
            test_labels = nny_test
            test_data = torch.from_numpy(test_data).float()
            test_labels = torch.from_numpy(test_labels).long()
            outputs = model(test_data)
            val, predicted = torch.max(outputs.data, 1)
            accuracy = 0
            for i in range(len(test_labels)):
                if is_neighbor(board_size, test_labels[i], predicted[i]):
                    accuracy += 1
                # if check_prediction(board_size, cluster, test_labels[i], predicted[i]):
                #     accuracy += 1
            accuracy = accuracy / test_labels.size(0)
            if accuracy > currenr_best:
                currenr_best = accuracy
                flag = True
                
            print(f"Test Accuracy: {accuracy:.4f}, Best Accuracy: {currenr_best:.4f}") 
                
        return currenr_best, flag

    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (epoch+1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            currenr_best_acc, flag = test(currenr_best_acc)
            if flag:
                current_best_model = copy.deepcopy(model)
                
    print('Best accuracy is :', currenr_best_acc)
    return current_best_model
    
    
    
        