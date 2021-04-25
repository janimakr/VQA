from keras.preprocessing.text import Tokenizer
from pickle import dump

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

# create a dictionary of the image:descriptions from the images selected for training and add a atart and end pointer: startseq,endseq 
def load_clean_descriptions(filename, dataset):
	doc=load_doc(filename)
	descriptions=dict()
	for line in doc.split('\n'):
		tokens=line.split()
		image_id,image_desc=tokens[0],tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id]=list()
			desc='startseq '+' '.join(image_desc) +' endseq'
			descriptions[image_id].append(desc)
	return descriptions

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

# load training dataset (6K images)
filename='D:\Professional\Project\Caption Generator\Flickr8k_text\Flickr_8k.trainImages.txt'
train=load_set(filename)
print('Dataset: %d' % len(train))
train_descriptions=load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer=create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))