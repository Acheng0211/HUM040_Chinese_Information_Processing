from pyAudioAnalysis import audioSegmentation as aS
import os

# 准备音频文件
audio_file = "/home/ares/hgj/NLP/ass2/speech_recognition/examples/microphone-results.wav"
root_dir = "/home/ares/hgj/NLP/ass2/pyAudioAnalysis/pyAudioAnalysis/data/models/"
# gt_file = audio_file.replace(".wav", ".segments")

# 使用pyAudioAnalysis进行情绪识别
    #sm
# [flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, root_dir + "svm_rbf_sm", "svm_rbf", True)
    #4class
[flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, root_dir + "svm_rbf_4class", "svm_rbf", True)
    #movie8class
# [flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, root_dir + "svm_rbf_movie8class", "svm_rbf")
    #speaker_10
# [flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, root_dir + "svm_rbf_speaker_10", "svm_rbf")
    #speaker_male_female
# [flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, root_dir + "svm_rbf_speaker_male_female", "svm_rbf")

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

