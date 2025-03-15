#!/usr/bin/env python
# coding: utf-8

# # **Upload and Run the Below Code in Google Collab**

# In[ ]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install mediapipe-model-maker')


# # **Import the Required Models**

# In[ ]:


from google.colab import files
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt


# # **Add the Downloaded dataset to your google drives `My Drive` Section. Then use Below Code to connect it to this Project**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


my_folder_path = '/content/drive/MyDrive/sign_language_dataset'


# In[ ]:


print(my_folder_path)


# In[ ]:


print(my_folder_path)
labels = []
for i in os.listdir(my_folder_path):
  if os.path.isdir(os.path.join(my_folder_path, i)):
    labels.append(i)
for label in labels:
  print(label + "\n")


# In[ ]:


print(len(labels))


# # **From Below Run All the Steps to Train and then Download the Trained Model**

# In[ ]:


NUM_EXAMPLES = 10

for label in labels:
  label_dir = os.path.join(my_folder_path, label)
  example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
  fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
  for i in range(NUM_EXAMPLES):
    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')

plt.show()


# In[ ]:


data = gesture_recognizer.Dataset.from_folder(
    dirname=my_folder_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)


# In[ ]:


hparams = gesture_recognizer.HParams(export_dir="exported_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)


# In[ ]:


loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")


# In[ ]:


model.export_model()
get_ipython().system('ls exported_model')


# In[ ]:


files.download('exported_model/gesture_recognizer.task')

