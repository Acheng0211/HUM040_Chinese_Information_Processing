from pyAudioAnalysis import audioSegmentation as aS
import os

# 准备音频文件
audio_file = "/home/ares/hgj/NLP/ass2/speech_recognition/examples/microphone-results.wav"

# 使用pyAudioAnalysis进行情绪识别
[flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, "/home/ares/anaconda3/envs/nlp/lib/python3.10/site-packages/pyAudioAnalysis/data/models/svm_rbf_sm", "svm_rbf")

# 定义情绪类别
emotion_classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# 输出结果
for i, flag in enumerate(flags_ind):
    # 将flag转换为整数
    int_flag = int(flag)
    print(f"Segment {i}: {emotion_classes[int_flag]}")
print(f"Classes: {classes_all}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {CM}")

