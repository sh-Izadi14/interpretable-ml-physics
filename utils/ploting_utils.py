import matplotlib.pyplot as plt

def plot_loss_curves(loss_fn_name, epochs_run, train_losses, test_losses):
        plt.title(f'{loss_fn_name} Curves')
        plt.plot(epochs_run, train_losses, c ='b', label='Train Loss')
        plt.plot(epochs_run, test_losses, c = 'r', label='Test Loss')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()


def plot_predictions(train_data,
                     train_labels,
                     test_data, 
                     test_labels, 
                     predictions=None):
    plt.scatter(train_data[:,0], train_labels, c='b', s=4, label="Training data")
    plt.scatter(test_data[:,0], test_labels,  c='g', s=4, label="test data")
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data[:,0], predictions, c="r", s=4, label="Predictions")
    
    plt.legend()