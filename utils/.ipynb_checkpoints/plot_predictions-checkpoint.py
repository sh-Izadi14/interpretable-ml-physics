import matplotlib.pyplot as plt

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