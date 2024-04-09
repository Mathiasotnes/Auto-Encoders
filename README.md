# Auto-Encoders
Deep learning auto-encoders used for generative models and anomaly detection. 

# Technologies
- Tensorflow

# Development plan

1. Start by implementing a way of representing MNIST data in lower dimention by training a encoder network using a 
convolutional neural network with stride > 1.

2. Create a decoder by using a transposed convolutional network, and use binary cross-entropy as loss.

3. Create anomaly detector by training the auto-encoder on the dataset using missing values, and visualize the top-k detected anomalies.

4. 


# Implementation status

### AE-BASIC X
Implement the autoencoder, learn from standard MNIST data, and show reconstruction results.

### AE-GEN X
Show results for the AE-as-a-generator task on MNIST data.

### AE-ANOM X
Show results for the AE-as-an-anomaly-detector task on MNIST data. Show the top-k anomalous examples from the test set.

### AE-STACK X
Show the results for the AE-GEN and AE-ANOM tasks when learn- ing from StackedMNIST data. Be prepared to discuss how you adapted the model structure when going from one to three color channels.

### VAE-BASIC X
Implement the variational autoencoder, learn from standard MNIST data, and show reconstruction results.

### VAE-GEN X
Show results for the VAE-as-a-generator task on MNIST data.

### VAE-ANOM X
Show results for the VAE-as-an-anomaly-detector task on MNIST data. NOTE! This is different from the AE-ANOM code. Simply doing the same as for the AE will give zero points.

### VAE-STACK X
Show the results for the VAE-GEN and VAE-ANOM tasks when learning from StackedMNIST data.