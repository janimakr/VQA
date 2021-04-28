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
def load_doc(filename):
	file=open(filename, 'r')
	text=file.read()
	file.close()
	return text
# from predefined set of 6k image names for training extract the names of the files and store in the list dataset 
def load_set(filename):
	doc=load_doc(filename)
	dataset=list()
	for line in doc.split('\n'):
		if len(line)<1:
			continue
		identifier=line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
# create a dictionary of the image:descriptions from the images selected for training and add a start and end pointer: startseq,endseq 
def load_clean_descriptions(filename, dataset):
	doc=load_doc(filename)
	descriptions=dict()
	for line in doc.split('\n'):
		tokens=line.split()
		image_id,image_desc=tokens[0],tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id]=list()
			desc='startseq '+' '.join(image_desc)+' endseq'
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features of the dataset element that are extracted by the VGG19
def load_photo_features(filename, dataset):
	all_features=load(open(filename, 'rb'))
	features={k: all_features[k] for k in dataset}
	return features
 
# covert a dictionary of clean descriptions to a list of descriptions, appending each description to a list
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
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
	X1,X2,y=list(),list(),list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq=tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq,out_seq=seq[:i], seq[i]
			# pad input sequence
			in_seq=pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq=to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1),array(X2),array(y)
 
# THE MODEL
def define_model(vocab_size, max_length):
	# feature extractor model for image-->expects a 4096 representation of the image which represents the image with a 256 element vector
	inputs1=Input(shape=(4096,))
	fe1=Dropout(0.5)(inputs1)
	fe2=Dense(256, activation='relu')(fe1)
	# sequence model for text-->Words are embedded first and then a LSTM is used to generate a 256 element output
	inputs2=Input(shape=(max_length,))
	se1=Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2=Dropout(0.5)(se1)
	se3=LSTM(256)(se2)
	# This layer uses the output of both the previous layers and generate and uses a relu action intially
	#  followed by softmax for generating final sequence of words
	decoder1=add([fe2, se3])
	decoder2=Dense(256, activation='relu')(decoder1)
	outputs=Dense(vocab_size, activation='softmax')(decoder2)
	model=Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.summary()
	return model
 
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions,photos,tokenizer,max_length,vocab_size):
	# loop for ever over images
	while 1:
		for key,desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img,in_seq,out_word=create_sequences(tokenizer,max_length,desc_list,photo,vocab_size)
			yield [in_img, in_seq],out_word
 
# load training dataset (6K images)
filename='D:\Jenny-work\Projects\Open Lab\Open Lab\Flickr8k_text\Flickr_8k.trainImages.txt'
train=load_set(filename)
print('Dataset: %d' % len(train))
train_descriptions=load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features=load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
tokenizer=create_tokenizer(train_descriptions)
vocab_size=len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length=max_length(train_descriptions)
print('Description Length: %d' % max_length)
 
# define the model's vocabulary size and max length
model=define_model(vocab_size, max_length)
epochs=20
# Fitting the model by progressive training of the model and saving the model after every epoch
steps=len(train_descriptions)
for i in range(epochs):
	generator=data_generator(train_descriptions, train_features,tokenizer,max_length, vocab_size)
	model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
	model.save('model_' + str(i) + '.h5')
