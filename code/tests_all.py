

# In[92]:

def varied_training(graph=graph,
                    training_data=training_data,
                    learning_rate=0.1,
                    batch_size=10,
                    test_data=test_data,
                    validation_data=validation_data,
                    epochs=10):
    
    loss = Euclidean()
    optimizer = SGD(learning_rate=learning_rate,
                    batch_size=batch_size)
    network = Network(graph)
    network.train(training_data = training_data,
                  loss = loss,
                  optimizer = optimizer,
                  test_data = test_data,
                  epochs = epoch)
    accuracy = network.test(validation_data)/len(validation_data)
    network.validation_error = 1-accuracy
    return(network)


# In[93]:

default_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

learning_rate = 3.0
batch_size = 10
epoch = 10

del network


# # Testing variations

# In[94]:

import time as time


# ### 01 : Layer variation

# In[95]:

graph_config_layer_variation = [
    [
    ("FullyConnected", {"shape": (10, 784)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (50, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 50)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (300, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (30, 300)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (500, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (100,500)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (30, 100)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (500, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (300,500)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (200,300)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (30, 200)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]
]


# In[96]:

default_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

learning_rate = 3.0
batch_size = 10
epoch = 25

variation_layer = []
for variation in range(5):
    
    print(variation)
    graph_config = graph_config_layer_variation[variation]
    start = time.time()
    current_model = varied_training(graph=Graph(graph_config),
                        training_data=training_data,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        test_data=test_data,
                        validation_data=validation_data,
                        epochs=epoch)
    stop = time.time()
    variation_layer += [[variation,current_model.test_error,current_model.validation_error,stop-start]]
    del current_model


# ### 02 : Node variation

# In[97]:

graph_config_node_variation=[
    [
    ("FullyConnected", {"shape": (500, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 500)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (300, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 300)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (200, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 200)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (100, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 100)}),
    ("Sigmoid", {})
],
    [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]
]


# In[98]:

learning_rate = 3.0
batch_size = 10
epoch = 25

variation_node = []
for variation in range(5):
    
    print(variation)
    graph_config = graph_config_node_variation[variation]
    start = time.time()
    current_model = varied_training(graph=Graph(graph_config),
                        training_data=training_data,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        test_data=test_data,
                        validation_data=validation_data,
                        epochs=epoch)
    stop = time.time()
    variation_node += [[variation,current_model.test_error,current_model.validation_error,stop-start]]
    del current_model


# ### 03 : Learning rate : 2 layers

# In[99]:

learning_rate_variation = [0.0001,0.001,0.01,0.1,1.0,10.0]


# In[ ]:

default_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

#learning_rate = 3.0
batch_size = 10
epoch = 25

variation_learning_rate = []
for variation in range(len(learning_rate_variation)):
    
    graph_config = default_config
    learning_rate = learning_rate_variation[variation]
    print(variation)
    start = time.time()
    current_model = varied_training(graph=Graph(default_config),
                                    training_data=training_data,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    test_data=test_data,
                                    validation_data=validation_data,
                                    epochs=epoch)
    stop = time.time()
    variation_learning_rate += [[variation,current_model.test_error,current_model.validation_error,stop-start]]
    del current_model


# ### 04 : Batch size variation : 2 layers

# In[ ]:

batch_size_variation = [1,10,25,50,100]


# In[ ]:

default_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

learning_rate = 3.0
#batch_size = 10
epoch = 25

variation_batch_size = []
for variation in range(5):
    
    graph_config = default_config
    batch_size = batch_size_variation[variation]
    print(variation)
    start = time.time()
    current_model = varied_training(graph=Graph(graph_config),
                                    training_data=training_data,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    test_data=test_data,
                                    validation_data=validation_data,
                                    epochs=epoch)
    stop = time.time()
    variation_batch_size += [[variation,current_model.test_error,current_model.validation_error,stop-start]]
    del current_model


# ### 05 : Epoch variation : 2 layers

# In[ ]:

epoch_variation = [1, 10, 25, 50, 100]


# In[ ]:

default_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

learning_rate = 3.0
batch_size = 10
#epoch = 25

variation_epoch = []
for variation in range(5):
    
    print(variation)
    graph_config = default_config
    epoch = epoch_variation[variation]
    start = time.time()
    current_model = varied_training(graph=Graph(graph_config),
                                    training_data=training_data,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    test_data=test_data,
                                    validation_data=validation_data,
                                    epochs=epoch)
    stop = time.time()
    variation_epoch += [[variation,current_model.test_error,current_model.validation_error,stop-start]]
    del current_model


# In[164]:

test_results = [variation_layer,variation_node,variation_learning_rate,variation_batch_size,variation_epoch]


filename = './results.pkl'

filehandler = open(filename, 'w') 
cPickle.dump(test_results, filehandler)
filehandler.close()


# # Report

# In[109]:

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


filename = './results.pkl'
filehandler = open(filename, 'r') 
cPickle.load(filehandler)


# ### 01. Variation of layers

# In[127]:

names = ['layers','error_train_byEpoch','validation_error_final','train_time']
results = variation_layer

data_plot=pd.DataFrame(results,columns=names)
data_plot['layers']+=1
data_plot


# ###### Observations :
# 
# * A trivial observation is that training time increases with increase in number of layers used, because of increase in number of weights to train for.
# 
# * Validation error drops dramatically with increasing number of layers (1-3), my reasoning being that the increased complexity of additional layers leads to better approximation of the output function.
# 
# * The increase in error might be because of overfitting or because of low number of updates. In the second case, since we have a large number of weights involved, a larger number of training epochs are warranted.
# 
# * Given the high accuracy with lower training time & lesser computational complexeity obtained by the networks with just 3 layers, that seems to be a better approach than training a deeper network for larger number epochs.

# In[143]:

sns.set_style('whitegrid')
plot = sns.plt.figure(figsize=(10,4))

plot.add_subplot(121)
sns.plt.plot(data_plot.layers,
             data_plot.validation_error_final)
sns.plt.xlabel('# Layers')
sns.plt.ylabel('Validation error')
sns.plt.title('Validation error vs No. of layers');

plot.add_subplot(122)
sns.plt.plot(data_plot.layers,
             data_plot.train_time)
sns.plt.xlabel('# Layers')
sns.plt.ylabel('Training time')
sns.plt.title('Training Time vs No. of layers');


# ### 02. Variation of nodes

# In[145]:

names = ['nodes','error_train_byEpoch','validation_error_final','train_time']
results = variation_node

data_plot=pd.DataFrame(results,columns=names)
data_plot['nodes'] = [500,300,200,100,30]
data_plot


# ###### Observations :
# 
# * A trivial observation is that training time increases with increase in number of nodes used, because of increase in number of weights to train for.
# 
# * Validation error is extremely low for low number of nodes in our setting, as a result of noisy combination formed when using high number of nodes in the hidden layer without appropriate training epochs. High number of nodes need longer to reach an optimal value and hence, in the current setting, their accuray is low.
# 
# * Given the high accuracy with lower training time & lesser computational complexeity obtained by the networks with just 30 neurons in hidden layer, that seems to be a better setting than training a wider network for larger number epochs.

# In[147]:

sns.set_style('whitegrid')
plot = sns.plt.figure(figsize=(10,4))

plot.add_subplot(121)
sns.plt.plot(data_plot.nodes,
             data_plot.validation_error_final)
sns.plt.xlabel('# Nodes')
sns.plt.ylabel('Validation error')
sns.plt.title('Validation error vs No. of nodes');

plot.add_subplot(122)
sns.plt.plot(data_plot.nodes,
             data_plot.train_time)
sns.plt.xlabel('# Nodes')
sns.plt.ylabel('Training time')
sns.plt.title('Training Time vs No. of nodes');


# ### 03. Variation of learning rate

# In[154]:

names = ['learning_rate','error_train_byEpoch','validation_error_final','train_time']
results = variation_learning_rate

data_plot=pd.DataFrame(results,columns=names)
data_plot['learning_rate'] = learning_rate_variation
data_plot


# ###### Observations :
# 
# * Training time decreases with increase in learning rate because our optimizer descends faster with larger rates. FOr small rates, the effect of dWs is extremely small & hence the optimizer has to run for a long time to actually make seizable improvements to the weights.
# 
# * Validation error is low for high learning rate in the given context because of bigger strides leading to faster updates towards the minima. In geenral, larger weights might overshoot the minima value & smaller weights might take a very long time to reach the minima value & so we have to find a balance between the two.

# In[158]:

sns.set_style('whitegrid')
plot = sns.plt.figure(figsize=(10,4))

plot.add_subplot(121)
sns.plt.plot(np.log10(data_plot.learning_rate),
             data_plot.validation_error_final)
sns.plt.xlabel('log(Learning rate)')
sns.plt.ylabel('Validation error')
sns.plt.title('Validation error vs Learning rate');

plot.add_subplot(122)
sns.plt.plot(np.log10(data_plot.learning_rate),
             data_plot.train_time)
sns.plt.xlabel('log(Learning rate)')
sns.plt.ylabel('Training time')
sns.plt.title('Training Time vs Learning rate');


# ### 04. Variation of batch size

# In[159]:

names = ['batch_size','error_train_byEpoch','validation_error_final','train_time']
results = variation_batch_size

data_plot=pd.DataFrame(results,columns=names)
data_plot['batch_size'] = batch_size_variation
data_plot


# ###### Observations :
# 
# * A trivial observation is that training time increases with increase in batch size, because of decrease in total updates required.
# 
# * Error decreases sharply on increasing batch size from 1 to 10 ( essentially, batch size 1 = SGD & batch size = 10 implies mini-batch gradient descent) because the gradients calcualted in this way are more stable. On further increase in batch size, error starts increasing because insufficient updates are done as the total size of input is just 10k.
# 
# * Given the high accuracy with batch_size =10, it seems to be the ideal choice.

# In[160]:

sns.set_style('whitegrid')
plot = sns.plt.figure(figsize=(10,4))

plot.add_subplot(121)
sns.plt.plot(data_plot.batch_size,
             data_plot.validation_error_final)
sns.plt.xlabel('Batch size')
sns.plt.ylabel('Validation error')
sns.plt.title('Validation error vs Batch size');

plot.add_subplot(122)
sns.plt.plot(data_plot.batch_size,
             data_plot.train_time)
sns.plt.xlabel('Batch size')
sns.plt.ylabel('Training time')
sns.plt.title('Training Time vs Batch size');


# ### 05. Variation of epochs

# In[161]:

names = ['epochs','error_train_byEpoch','validation_error_final','train_time']
results = variation_epoch

data_plot=pd.DataFrame(results,columns=names)
data_plot['epochs'] = epoch_variation
data_plot


# ###### Observations :
# 
# * A trivial observation is that training time increases is a directly proportional to epochs, because the train method is nalled n(epochs) times.
# 
# * Large value for epochs generally lead to lower error rate on validation set. This is because of more optimisation runs in the train phase.
# 
# * For a given problem, a reasonable amount of epochs can be decided by checking difference in accuracy for each successive epoch & stopping when the accuracy-gain is below a pre-defined threshold.

# In[163]:

sns.set_style('whitegrid')
plot = sns.plt.figure(figsize=(10,4))

plot.add_subplot(121)
sns.plt.plot(data_plot.epochs,
             data_plot.validation_error_final)
sns.plt.xlabel('# epochs')
sns.plt.ylabel('Validation error')
sns.plt.title('Validation error vs No. of epochs');

plot.add_subplot(122)
sns.plt.plot(data_plot.epochs,
             data_plot.train_time)
sns.plt.xlabel('# epochs')
sns.plt.ylabel('Training time')
sns.plt.title('Training Time vs No. of epochs');
