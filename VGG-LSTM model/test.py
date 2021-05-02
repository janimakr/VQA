from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model
import sys


 
# extract features from the input image using vgg19
def extract_features(file):
	model=VGG19()
	model=Model(inputs=model.inputs, outputs=model.layers[-2].output)
	image=load_img(file, target_size=(224, 224))
	image=img_to_array(image)
	image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image=preprocess_input(image)
	feature=model.predict(image, verbose=0)
	return feature
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word,index in tokenizer.word_index.items():
		if index==integer:
			return word
	return None
 
# generate a description for an image by generating the words sequentially from the probability
def generate_desc(model, tokenizer, photo, max_length):
	in_text='startseq'
	for i in range(max_length):
		sequence=tokenizer.texts_to_sequences([in_text])[0]
		sequence=pad_sequences([sequence], maxlen=max_length)
		yhat=model.predict([photo,sequence], verbose=0)
		yhat=argmax(yhat)
		word=word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text+=' '+word
		if word=='endseq':
			break
	return in_text
 

tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
# intiating prediction
model=load_model('model_13.h5')
photo=extract_features('testinput3.jpg')
description=generate_desc(model, tokenizer, photo, max_length)
print("Description generated: ",description)
with open('outputfromDescriptionModel.txt', 'w') as f:
    sys.stdout = f 
    print(description[9:len(description)-7])

