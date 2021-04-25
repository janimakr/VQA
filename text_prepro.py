import string
# extracting the contents of a file and returning to the user
def load_doc(filename):
	file=open(filename, 'r')
	text=file.read()
	file.close()
	return text
 
# for each image traverse through the document and create a dictionary where, key is image id and the values are the descriptions of the image(5 descriptions per image), descriptions are stored as a list 
def load_descriptions(doc):
	mapping=dict()
	for line in doc.split('\n'):
		tokens=line.split()
		if len(line) < 2:
			continue
		image_id,image_desc=tokens[0],tokens[1:]
		image_id=image_id.split('.')[0]
		image_desc=' '.join(image_desc)
		if image_id not in mapping:
			mapping[image_id]=list()
		mapping[image_id].append(image_desc)
	return mapping

# picking each word in the descriptions and cleaning the text by converting all to lower case,removing the punctuations and single character words like 's','a' etc.
def clean_descriptions(descriptions):
	table=str.maketrans('','',string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc=desc_list[i]
			desc=desc.split()
			desc=[word.lower() for word in desc]
			desc=[w.translate(table) for w in desc]
			desc=[word for word in desc if len(word)>1]
			desc=[word for word in desc if word.isalpha()]
			desc_list[i]=' '.join(desc)
 
# convert the loaded descriptions into a vocabulary of words
# identifying the corpus/dictionary for the application
def to_vocabulary(descriptions):
	all_desc=set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
 
# save descriptions to file, descripion per line, i.e. image id followed by one of the descriptions of the image
def save_descriptions(descriptions, filename):
	lines=list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data='\n'.join(lines)
	file=open(filename, 'w')
	file.write(data)
	file.close()
 
filename='D:\Professional\Project\Caption Generator\Flickr8k_text\Flickr8k.token.txt'
doc=load_doc(filename)
descriptions=load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
clean_descriptions(descriptions)
vocabulary=to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
save_descriptions(descriptions, 'descriptions.txt')