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
 To run the Visual Attention model
  - Clone the repo 
  - Run the VQA_with_caption.ipynb file in your local system(Jupyter) or using Google colab if you dont have a GPU
