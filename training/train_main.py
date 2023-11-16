from training_function import train

#give the training scheme a name that reflects the training scheme
#this will be in the name_of_training_schema of the model and the logfile
name_of_training_schema = "1xalldata_10epochs_all_equal_new"
#Parameters that can and should be changed
max_length = 128
model_name = 'google/mobilebert-uncased'
num_epochs = 10
batch_number = 100
batch_size=32
#settings = ((20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),
 #           (20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20))
settings = ((1,0,0,0,0),(1,0,0,0,0),(1,0,0,0,0),(1,0,0,0,0),(1,0,0,0,0),
            (1,0,0,0,0),(1,0,0,0,0),(1,0,0,0,0),(1,0,0,0,0),(1,0,0,0,0))
train(name_of_training_schema = name_of_training_schema, max_length = max_length, model_name = model_name,
          num_epochs = num_epochs, batch_number = batch_number, batch_size=batch_size,
          settings = settings)