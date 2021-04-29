from keras.preprocessing.text import Tokenizer
from pickle import dump

# extracting the contents of a file and returning to the user
def loadFile(Fname):
	f=open(Fname, 'r')
	txt=f.read()
	f.close()
	return txt

#  extract the names of the files
#  and store in the list dataset 
def loading(Fname):
	doc=loadFile(Fname)
	dset=list()
	for perline in doc.split('\n'):
		if len(perline)<1:
			continue
		id=perline.split('.')[0]
		dset.append(id)
	return set(dset)

# create a dictionary of the image:descriptions from the images selected for 
# training and add a atart and end pointer: startseq,endseq 
def loadCleanDescriptions(Fname, dset):
	doc=loadFile(Fname)
	desc=dict()
	for perline in doc.split('\n'):
		token=perline.split()
		imgid,img_desc=token[0],token[1:]
		if imgid in dset:
			if imgid not in desc:
				desc[imgid]=list()
			desc='startseq '+' '.join(img_desc) +' endseq'
			desc[imgid].append(desc)
	return desc


#  appending each description to a list
def toLines(desc):
	allDesc=list()
	for key in desc.keys():
		[allDesc.append(d) for d in desc[key]]
	return allDesc

# fit a tokenizer on the descriptions and maintain tokenization stats
def Tokenisation(desc):
	l=toLines(desc)
	tizer=Tokenizer()
	tizer.fit_on_texts(l)
	return tizer


Fname='D:\Jenny-work\Projects\Open Lab\Open Lab\Flickr8k_text\Flickr_8k.trainImages.txt'
train=loading(Fname)
print('Dataset: %d' % len(train))
train_desc=loadCleanDescriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_desc))
tizer=Tokenisation(train_desc)
dump(tizer, open('tokenizer.pkl', 'wb'))