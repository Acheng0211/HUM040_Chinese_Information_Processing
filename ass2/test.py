# import speech_recognition as sr

# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Say something!")
#     audio = r.listen(source)
 
# test = sr.AudioFile('/home/ares/hgj/NLP/ass2/speech_recognition/examples/english.wav')
 
# with test as source:
#     audio = r.record(source)
 
# said = r.recognize_google(audio, language='en-US')
# print("google think you said:",said)

from pyannote.audio import Pipeline

# Initialize the pipeline
pipeline = Pipeline.from_pretrained("/home/ares/hgj/NLP/ass2/pyannote-speaker-diarization-3.1/config.yml")

# Apply the pipeline to an audio file
diarization = pipeline("/home/ares/hgj/NLP/ass2/speech_recognition/examples/english.wav")

# Print the result
for segment, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{segment.start:.1f}s - {segment.end:.1f}s: speaker {speaker}")
