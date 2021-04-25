from os import listdir
from pickle import dump
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
 
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model=VGG19()
	# re-structure the model to remove the last layer of the model
	model=Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# summarize displaying model paramenters
	print(model.summary())
	# traverse through all the images,extract features from each photo and store in a dictionary
	features=dict()
	for name in listdir(directory):
		# load an image from file
		filename=directory+'/'+name
		image=load_img(filename,target_size=(224, 224))
		# convert the image pixels to a numpy array
		image=img_to_array(image)
		# reshape data for the model
		image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model trained for imagenet data
		image=preprocess_input(image)
		# get features
		feature=model.predict(image, verbose=0)
		# get image id (name of the image without extention)
		image_id=name.split('.')[0]
		# store feature
		features[image_id]=feature
		print('>%s' % name)
	return features
# extract features from all images
directory='D:\Jenny-work\Projects\Open Lab\Open Lab\Flicker8k_Dataset'
features=extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features,open('features.pkl', 'wb'))
