#Traina  binary classifier to perform sentiment analysis on an IMDB dataset

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.__version__)


#Downloading dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

#Prints files and subdirectories
print(os.listdir(dataset_dir))

#Creates a directory path to the subdirectory called 'train' that is within original directory, next line prints 'train' subdirectories
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

#Creating directory path to an exmaple text file and printing it out
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())

#Creating a directory path for 'unsup' and then removing it
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)



batch_size = 32
seed = 42
#This function is going in and splitting our training data into 20% validation and 80% testing(stored in raw_train_ds), bathsize is 32
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

#printing out a few example reviews and the actual results(0 is negative review, 1 is a positive review)
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])


#We are splitting the same data as above, but now assigning validation data to raw_val_ds
#By using the same seed as before, we are assuring that their is no overalp between training and validation data, we could've also turned shuffle to false for both calls, and avoided seed
#Using the smae seed number starts the random number generating to start at the same place, so it is replicated from before.
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)


#Before, we were splitting the training data form 80 percent train and 20 percent validate, NOW we are creating a directory path from a whole differnt file
#Because this is a different file, that is why we dont have to worry about splitting, randomizing, subset, or overlapping data, this whole file is test data
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)


#Standardize data: Cleaning text data(remove puctuation and HTML eleements)
#Tokenize data: Splitting text into smaller fragments like words or chaarcters(Split on whitespace)
# Vectorize data: Turn data into a numerical format for machine elarning model(Covert tokens into numbers)

 #Our normal TextVectorization  layer will not remove html tags, so we have to write our ownn custom function to do so
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

#Now since we got rid of HTML eleemnts we canc reate our TextVectorization layer to standardize, tokenize, and vectorize our data

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization, #Calls our prebuilt function to clean data
    max_tokens=max_features, #Keeps top 10000 tokens
    output_mode='int', #Our goal is to make our data numerical for training
    output_sequence_length=sequence_length) #length of the output sequences


# Make a text-only dataset (without labels), then call adapt
#We are taking purely our text data here, without the labels for training
train_text = raw_train_ds.map(lambda x, y: x)
#Dont understand this line fully but we are "preparing" the TextVectorization layer to tokenize and preprocess text data when you use it later
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label



# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))


#Checking the token at 2 different indexes
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

#Now we are actually vectorizing our data for the machine learning model
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


#Finds optimal buffersize
AUTOTUNE = tf.data.AUTOTUNE

#.cache() helps reduce intial data loading time
#.prefetch helps keep the model busy while the next set of data is prepared
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)



embedding_dim = 16

#Building the text classification model
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim), #First layer of model converts token sequences to dense vectors
  layers.Dropout(0.2), #Second layer used to avoid overfitting(prevents model from relaince on a single word embeding, by inserting randomness by turning neurons to 0 as the data is training)
  layers.GlobalAveragePooling1D(), #Dont fully understand, but it reduces overfitting and also reduces spatial dimensions while keeping global structure
  layers.Dropout(0.2), #Helps prevent model from making potential decisions based on noisy information
  layers.Dense(1)]) # Makes the actual classification between 0 and 1

model.summary()


model.compile(loss=losses.BinaryCrossentropy(from_logits=True), # This is a type of loss function for binary values(0 or 1) and logits is saying that this is raw data
              optimizer='adam', # This is our optimization function
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0)) # we are using binary accuracy(checks if model is right or not)


#Since we have built and compiled our model, it is time for training
epochs = 10
# model.fit initiates the training process and fitsd the model with our training data
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs) # How many times the model will run through the traaining data. Separates into smaller batches which updates model after each batch by calculating loss and optimizing.




#Evaluates the model and finds the average loss and accuracy of the model
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

#Line retrieves a dictionary containing the history of the model during training
history_dict = history.history
history_dict.keys() #We can use these keys for plotting the effectiveness of our graph: loss, binary_accuracy, val_loss, val_binary_accuracy



#Separates data from dictionary into its own elements
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

#Creates a range to display epochs
epochs = range(1, len(acc) + 1)


# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')

#Labeling graph
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#Allows us to distinguish what blue line represents(Validation loss) and what the blue dot means(Training loss)
plt.legend()

plt.show()



#Plotting training and validation accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

#We can see that the training data accuracy keeps increasing while the validation begins to flatten out and peek before the training data. This is due to overfitting since the model is use to the training data
 







