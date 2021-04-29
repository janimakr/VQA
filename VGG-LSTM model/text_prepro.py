import string
# extracting the contents of a file and returning to the user  -text in token form 
def get_tokens(datasettokens):
	file=open(datasettokens, 'r')
	tokens=file.read()
	file.close()
	return tokens
 
# for each image traverse through the document and create a dictionary where,
#  key is image id and the values are the descriptions of the image
def create_dictionary(document):
	connect=dict()
	for line in document.split('\n'):
		tokens=line.split()
		if len(line) < 2:
			continue
		img_id,img_description=tokens[0],tokens[1:]
		img_id=img_id.split('.')[0]
		img_description=' '.join(img_description)
		if img_id not in connect:
			connect[img_id]=list()
		connect[img_id].append(img_description)
	return connect

# picking each word in the descriptions and cleaning the text by converting all to lower case,
# removing the punctuations and single character words like 's','a' etc.
def clean_descriptions(descriptions):
	table=str.maketrans('','',string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc=desc_list[i]
			desc=desc.split()
			desc=[word.lower() for word in desc]
			desc=[word for word in desc if len(word)>1]
			desc=[word for word in desc if word.isalpha()]
			desc_list[i]=' '.join(desc)
 
# convert the loaded descriptions into a vocabulary of words
# identifying the corpus/dictionary for the application
def vocab_of_words(descriptions):
	all_desc=set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
 
# save descriptions to file,1 descripion per line, i.e. image id followed by one of the descriptions of the image
def save_descriptions(descriptions, filename):
	lines=list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data='\n'.join(lines)
	file=open(filename, 'w')
	file.write(data)
	file.close()
 
datasettokens='D:\Jenny-work\Projects\Open Lab\Open Lab\Flickr8k_text\Flickr8k.token.txt'
tokens=get_tokens(datasettokens)
descriptions=create_dictionary(tokens)
print('Loaded Descriptions: %d ' % len(descriptions))
clean_descriptions(descriptions)
vocab=vocab_of_words(descriptions)
print('Vocabulary Size: %d' % len(vocab))
save_descriptions(descriptions, 'descriptions.txt')