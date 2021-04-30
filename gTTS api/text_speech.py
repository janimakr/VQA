#pip install gTTS ( google Text to speech api before running this makkale)
from gtts import gTTS
from IPython.display import Audio
 answer="This is a trial audio used for testing. If you've run this, comment this part to get the answer for your question" # comment this part !
 # directly run the variable with the answer into the caption below 
tts = gTTS(answer)
tts.save('1.wav')
sound_file= '1.wav'
Audio(sound_file, autoplay=True)