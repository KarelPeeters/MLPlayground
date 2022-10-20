import whisper

print("Loading model")
model = whisper.load_model("base")

print("Loading audio")
audio = whisper.load_audio("count30.mp3")

print(audio.shape)

audio = whisper.pad_or_trim(audio)

print(audio.shape)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

print("Detecting language")
# detect the spoken language
_, probs = model.detect_language(mel)
print(f"All languages: {probs}")
print(f"Detected language: {max(probs, key=probs.get, )}")

print("Decoding audio")
# decode the audio
options = whisper.DecodingOptions(language="en")
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
