from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
 
# extracting the contents of a file and returning to the user
def loading(file):
	file=open(file, 'r')
	loaded=file.read()
	file.close()
	return loaded

# from predefined set of 6k image names for training extract the names of the files and store in the list dataset 
def load_tolist(file):
	document=loading(file)
	data=list()
	for line in document.split('\n'):
		if len(line)<1:
			continue
		identifier=line.split('.')[0]
		data.append(identifier)
	return set(data)
 
# create a dictionary of the image:descriptions from the images selected for training and add a start and end pointer: startseq,endseq 
def load_clean_descriptions(file, dataset):
	document=loading(file)
	descriptions=dict()
	for line in document.split('\n'):
		tokens=line.split()
		image_id,image_desc=tokens[0],tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id]=list()
			desc='startseq '+' '.join(image_desc)+' endseq'
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features of the dataset element that are extracted by the VGG19
def load_photo_features(file, dataset):
	all_features=load(open(file, 'rb'))
	features={k: all_features[k] for k in dataset}
	return features
 
# convert a dictionary of clean descriptions to a list of descriptions, appending each description to a list
def to_lines(descriptions):
	all_desc=list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer on the descrptions and maintain tokenisation stats
def create_tokenizer(descriptions):
	lines=to_lines(descriptions)
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# identify the longest description
def max_length(descriptions):
	lines=to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab):
	X1,X2,y=list(),list(),list()
	for desc in desc_list:
		seq=tokenizer.texts_to_sequences([desc])[0]
		for i in range(1, len(seq)):
			in_seq,out_seq=seq[:i], seq[i]
			in_seq=pad_sequences([in_seq], maxlen=max_length)[0]
			out_seq=to_categorical([out_seq], num_classes=vocab)[0]
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1),array(X2),array(y)
 
# THE MODEL
def define_model(vocab, max_length):
	inputs1=Input(shape=(4096,))
	fe1=Dropout(0.5)(inputs1)
	fe2=Dense(256, activation='relu')(fe1)
	inputs2=Input(shape=(max_length,))
	se1=Embedding(vocab, 256, mask_zero=True)(inputs2)
	se2=Dropout(0.5)(se1)
	se3=LSTM(256)(se2)
	# This layer uses the output of both the previous layers and generate and uses a relu action intially
	#  followed by softmax for generating final sequence of words
	decoder1=add([fe2, se3])
	decoder2=Dense(256, activation='relu')(decoder1)
	outputs=Dense(vocab, activation='softmax')(decoder2)
	model=Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.summary()
	return model
 
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions,photos,tokenizer,max_length,vocab):
	while 1:
		for key,desc_list in descriptions.items():
			photo = photos[key][0]
			in_img,in_seq,out_word=create_sequences(tokenizer,max_length,desc_list,photo,vocab)
			yield [in_img, in_seq],out_word
 
# load training dataset (6K images)
filename='D:\Jenny-work\Projects\Open Lab\Open Lab\Flickr8k_text\Flickr_8k.trainImages.txt'
train=load_tolist(filename)
print('Dataset: %d' % len(train))
train_descriptions=load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features=load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
tokenizer=create_tokenizer(train_descriptions)
vocab=len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab)
max_length=max_length(train_descriptions)
print('Description Length: %d' % max_length)
 
# define the model's vocabulary size and max length
model=define_model(vocab, max_length)
epochs=20
steps=len(train_descriptions)
for i in range(epochs):
	generator=data_generator(train_descriptions, train_features,tokenizer,max_length, vocab)
	model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
	model.save('model_' + str(i) + '.h5')
