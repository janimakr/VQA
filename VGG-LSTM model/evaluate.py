from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
 
# load the test images into memory as read only 
def load_test_img(filename):
	file=open(filename, 'r')
	text=file.read()
	file.close()
	return text
 
# load list of photo identifiers to process them line by line to a list and get img id 
def load_img_id(filename):
	document=load_test_img(filename)
	data=list()
	for line in document.split('\n'):
		if len(line)<1:
			continue
		identifier=line.split('.')[0]
		data.append(identifier)
	return set(data)
 
# load clean descriptions into memory
def load_clean_descriptions(file, data):
	document=load_test_img(file)
	descriptions=dict()
	for line in document.split('\n'):
		tokens=line.split()
		img_id,img_description=tokens[0],tokens[1:]
		if img_id in data:
			if img_id not in descriptions:
				descriptions[img_id]=list()
			desc='startseq '+' '.join(img_description)+' endseq'
			descriptions[img_id].append(desc)
	return descriptions
 
# load photo features and filter to only those present in dataset
def load_photo_features(file, data):
	full_features=load(open(file, 'rb'))
	features={k: full_features[k] for k in data}
	return features
 
# convert a dictionary of clean descriptions to a list of descriptions
def to_list(descriptions):
	all_descriptions=list()
	for key in descriptions.keys():
		[all_descriptions.append(d) for d in descriptions[key]]
	return all_descriptions
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines=to_list(descriptions)
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the length of the description with the most words
def max_length(descriptions):
	lines=to_list(descriptions)
	return max(len(d.split()) for d in lines)
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index==integer:
			return word
	return None
 
# generate a description for an image by iterating full length
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
 
# evaluate the skill of the model by storing actual and predicted
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual,predicted=list(),list()
	for key, desc_list in descriptions.items():
		yhat=generate_desc(model, tokenizer, photos[key], max_length)
		references=[d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
 

filename='D:\Jenny-work\Projects\Open Lab\Open Lab\Flickr8k_text\Flickr_8k.trainImages.txt'
train=load_img_id(filename)
print('Dataset: %d'%len(train))
train_descriptions=load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer=create_tokenizer(train_descriptions)
vocab_size=len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length=max_length(train_descriptions)
print('Description Length: %d' % max_length)

filename='D:\Jenny-work\Projects\Open Lab\Open Lab\Flickr8k_text\Flickr_8k.testImages.txt'
test=load_img_id(filename)
print('Dataset: %d' % len(test))
test_descriptions=load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
test_features=load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
 
# load the model and evaluate
filename='model_13.h5'
model=load_model(filename)
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)