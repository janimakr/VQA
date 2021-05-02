#pip install gTTS ( google Text to speech api before running this makkale)
from gtts import gTTS
from IPython.display import Audio
#  answer="This is a trial audio used for testing. If you've run this, comment this part to get the answer for your question" # comment this part !
# directly run the variable with the answer into the caption below 
with open('outputfromVQA.txt','r') as file:
    answer=file.read()
tts = gTTS(answer)
tts.save('finalaudio.wav')
sound_file= 'finalaudio.wav'
Audio(sound_file, autoplay=True)
print("\nPlay the audio file 'final audio' to listen! ")
