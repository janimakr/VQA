from os import listdir
from pickle import dump
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
 
# extract features from each photo in the directory to a create a dictionary where the image id is the key and values the features of the image. Intially load the model without the last layer , traverse through each image, convert them to an array and transform them to the image format of imagenet and their corresponding features are stored one after the another in a file called features.pkl(pickle file to seraialise objects)
def extract_features(directory):
	model=VGG19()
	model=Model(inputs=model.inputs, outputs=model.layers[-2].output)
	print(model.summary())
	features=dict()
	for name in listdir(directory):
		filename=directory+'/'+name
		image=load_img(filename,target_size=(224, 224))
		image=img_to_array(image)
		image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image=preprocess_input(image)
		feature=model.predict(image,verbose=0) #verbose valus tells what needs to be displayed during prediction
		image_id=name.split('.')[0]
		features[image_id]=feature
		print('>%s' % name)
	return features

# extract features from all images
directory='D:\Professional\Project\Caption Generator\Flickr8k_Dataset\Flicker8k_Dataset'
features=extract_features(directory)
print('Extracted Features: %d' % len(features))
dump(features,open('features.pkl', 'wb'))
