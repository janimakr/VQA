from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from os import listdir
from keras.models import Model
from pickle import dump
 
# feature extraction of images in the directory to a create a dictionary where the image id is the key and values the features of the image. Intially load the model without the last layer , traverse through each image, convert them to an array and transform them to the image format of imagenet and their corresponding features are stored one after the another in a file called features.pkl(pickle file to seraialise objects)
def featureExtraction(dir):
	model=VGG19()
	model=Model(inputs=model.inputs,outputs=model.layers[-2].output)
	print(model.summary())
	feature=dict()
	for fname in listdir(dir):
		filefname=dir+'/'+fname
		image=load_img(filefname,target_size=(224,224))
		image=img_to_array(image)
		image=image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
		image=preprocess_input(image)
		feature=model.predict(image,verbose=0) #verbose valus tells what needs to be displayed during prediction
		imgid=fname.split('.')[0]
		feature[imgid]=feature
		print('>%s' % fname)
	return feature

# extract features from all images and store in pkl file
dir='path\Flickr8k_Dataset\Flicker8k_Dataset'
feature=featureExtraction(dir)
print('Extracted Features: %d' % len(feature))
dump(feature,open('features.pkl', 'wb'))
