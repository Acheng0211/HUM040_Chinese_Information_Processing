import os
import numpy as np
import speech_recognition as sr
from pyAudioAnalysis import audioSegmentation as aS

# 定义音频文件路径
audio_file = "microphone-results.wav"

# 使用speech_recognition库录制音频
def record_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("请说话...")
        audio = recognizer.listen(source)
        print("录音完毕")

        # 保存音频文件
        with open(audio_file, "wb") as f:
            f.write(audio.get_wav_data())

# 调用函数录制音频
record_audio(audio_file)

# 使用pyAudioAnalysis进行情绪识别
[flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, "/home/ares/anaconda3/envs/nlp/lib/python3.10/site-packages/pyAudioAnalysis/data/models/svm_rbf_sm", "svm_rbf")

# 定义情绪类别
emotion_classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# 输出分类结果
for i, flag in enumerate(flags_ind):
    # 将flag转换为整数
    int_flag = int(flag)
    print(f"Segment {i}: {emotion_classes[int_flag]}")

print(f"Classes: {classes_all}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {CM}")
