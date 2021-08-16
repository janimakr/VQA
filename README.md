# **Visual QnA model**
Historically, building a system that can answer natural language questions about any image has been considered a very ambitious goal. Imagine a system that, given the image below, could answer these questions:

- >What is in the image?
- >Are there any humans?
- >What is the color of the ball?
- >Who has the ball?
- >How many dogs are in the image?




<img src="images/readme.jpg" alt="readme">

In this Repo we have tried to implement the VQA model  using  two architectures, 
- Vgg-19 and LSTM 
- Visual Attention Model 

Workflow:

- >A description is generated using the above architectures.
- >Input questions are answered using a pretrained model from Transformers library.
- >The output generated is converted from text-to-speech using the gTTS library.


Datasets: 
For [Visual Attention](http://cocodataset.org/#home) 
For [Vgg-19](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip), [LSTM](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

## How to run 
 To run the Visual Attention model:
  - Clone the repo. 
  - Run the VQA_with_caption.ipynb file in your local system(Jupyter) or using Google colab if you dont have a GPU.
 
 To run the Vgg19-LSTM model:
  - Clone the repo.
  - Download the dataset and store it in the same folder. (for [Vgg-19](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip), [LSTM](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip))
  - Run the image_prepro.py and text_prepro.py files in your local system to preprocess the dataset.
  - Run generate.py, followed by train.py to train the model to start generating descriptions.
  - To evaluate the results of training, run evaluate.py.
  - Test the model by giving the .h5 file (generated after training the model) and a sample image as inputs in the code.
  - To answer questions using generated description, run QnA.py and give any question as input.
  - To convert the obtained answer to speech (audio format), run text_speech.py.



 ## Individual modules were taken/referenced from from: 
 Caption generator for VGG19-LSTM Module:
  - https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
 <br/>Question Answering using pretrained transformer: 
  - https://towardsdatascience.com/question-answering-with-pretrained-transformers-using-pytorch-c3e7a44b4012
  <br/>Text to Speech(using gtts):
  - https://www.geeksforgeeks.org/convert-text-speech-python/ 
  
