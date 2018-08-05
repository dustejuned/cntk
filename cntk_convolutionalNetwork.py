# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:58:09 2018

@author: adm_dustej
"""

import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import sys
import cntk as C
from cntk.cntk_py import square
from cntk.internal import sanitize_input
from PIL import Image

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

input_dim_model = (1, 28, 28)
input_dim = 784
num_output_classes = 10
learning_rate = 0.2

minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
num_hidden_layers = 2
hidden_layer_dim = 400
    
def loadData(src, cimg):    
    print('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            if n[0] != 0x3080000:
                raise Exception('Invalid File: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))[0]
            
            if n != cimg:
                raise Exception('Invalid File: expected {0} entries.'.format(cimg))
            
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            
            if crow != 28 or ccol != 28:
                raise Exception('Invalid File: expected 28 rows/cols per image.')
                
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)    
            #print(res)
    finally:
         os.remove(gzfname)       
    return res.reshape((cimg, crow*ccol))

def loadLabels(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            if n[0] != 0x1080000:
                raise Exception('Invalid File: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))[0]
            
            if n != cimg:
                raise Exception('Invalid File: expected {0} entries.'.format(cimg))
                        
                
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)    
            #print(res)
    finally:
         os.remove(gzfname)       
    return res.reshape((cimg, 1))
    

def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    #print('data is:')
    #print(data)
    labels = loadLabels(labelsSrc, cimg)
    #print('labels are:')
    #print(labels)
    return np.hstack((data, labels)) 

# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving", filename )
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])               
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)

def create_reader(path, is_training, input_dim, num_label_classes):
    labelStream = C.io.StreamDef(field="labels", shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(field="features", shape=input_dim, is_sparse=False)
    
    deserializer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))
    
    return C.io.MinibatchSource(deserializer, randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

def create_model(features):
    with C.layers.default_options(init = C.glorot_uniform(), activation = C.ops.relu):
        h = features
        h = C.layers.Convolution2D(filter_shape = (5,5),
                                   num_filters = 8,
                                   strides = (1,1),
                                   pad = True,
                                   name = 'first_conv'
                                   )(h)
        h = C.layers.AveragePooling(filter_shape = (5,5), strides = (2, 2))(h)
        h = C.layers.Convolution2D(filter_shape = (5,5),
                                   num_filters = 16,
                                   strides = (1,1),
                                   pad = True,
                                   name = 'second_conv'
                                   )(h)
        h = C.layers.AveragePooling(filter_shape = (5,5), strides = (2, 2))(h)
        r = C.layers.Dense(num_output_classes, activation = None, name = 'classify')(h)        
        return r
    

def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))

    return mb, training_loss, eval_error
    
# URLs for the train image and label data
url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
num_train_samples = 60000

print("Downloading train data")
train = try_download(url_train_image, url_train_labels, num_train_samples)

# URLs for the test image and label data
url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
num_test_samples = 10000

print("Downloading test data")
test = try_download(url_test_image, url_test_labels, num_test_samples)

# Plot a random image
#sample_number = 3000
#plt.imshow(train[sample_number,:-1].reshape(28,28), cmap="gray_r")
#plt.axis('off')
#print("Image Label: ", train[sample_number,-1])

# Save the train and test files (prefer our default path for the data)
data_dir = os.path.join("..", "Examples", "Image", "DataSets", "MNIST")
if not os.path.exists(data_dir):
    data_dir = os.path.join("data", "MNIST")

print ('Writing train text file...')
savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)

print ('Writing test text file...')
savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)

print('Done')


input = C.input_variable(input_dim_model)
label = C.input_variable(num_output_classes)
normalize_input = input/255.0
squared_input = C.square(input/255.0)
sqrt_input = C.sqrt(input/255.0)


z = create_model(normalize_input)

print('Output shape of the first convolution layers:', z.first_conv.shape)
print('Output shape of the first convolution layers:', z.classify.b.value)

C.logging.log_number_of_parameters(z)

loss = C.cross_entropy_with_softmax(z, label)

label_error = C.classification_error(z, label)

lr_schedule = C.learning_parameter_schedule(learning_rate)

learner = C.sgd(z.parameters, lr_schedule)

trainer = C.Trainer(z, (loss, label_error), [learner])


data_found = False

for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
                 os.path.join("data", "MNIST")]:
    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found = True
        break

if not data_found:
    raise ValueError("Please generate the data by completing CNTK 103 Part A")

print("Data directory is {0}".format(data_dir))


# Create the reader to training data set
reader_train = create_reader(train_file, True, input_dim, num_output_classes)

# Map the data streams to the input and labels.
input_map = {
    label  : reader_train.streams.labels,
    input  : reader_train.streams.features
}

# Run the trainer on and perform model training
training_progress_output_freq = 500

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):

    # Read a mini batch from the training data file
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)

    trainer.train_minibatch(data)
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)
        
        
# Compute the moving average loss to smooth out the noise in SGD
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])


plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()        


# Read the training data
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}

# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0

for i in range(num_minibatches_to_test):

    # We are loading test data in batches specified by test_minibatch_size
    # Each data point in the minibatch is a MNIST digit image of 784 dimensions
    # with one pixel per dimension that we will encode / decode with the
    # trained model.
    data = reader_test.next_minibatch(test_minibatch_size,
                                      input_map = test_input_map)

    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


out = C.softmax(z)


# Read the data for evaluation
#reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

eval_minibatch_size = 25
#eval_input_map = {input: reader_eval.streams.features}

data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)


img_label = data[label].asarray()
img_data = data[input].asarray()
# reshape img_data to: M x 1 x 28 x 28 to be compatible with model
img_data = np.reshape(img_data, (eval_minibatch_size, 1, 28, 28))
predicted_label_prob = [out.eval(img_data[i]) for i in range(len(img_data))]


# Find the index with the maximum value for both predicted as well as the ground truth
pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]

print("Label    :", gtlabel[:25])
print("Predicted:", pred)

path = "data/"+"MysteryNumberD.bmp"

inp= Image.open(path).convert('L')

img= np.array(inp)

plt.imshow(img, cmap="gray_r")

plt.axis('off')

img_feature = img.flatten()

img_feature = img_feature/255.0

img_feature = np.reshape(img_feature, (1, 28, 28))

predicted_label_prob = out.eval(img_feature)


        
print(np.argmax(predicted_label_prob))