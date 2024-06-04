import os
import speech_recognition as sr
from pyAudioAnalysis import audioSegmentation as aS

# 定义录音保存路径
audio_file = "microphone-results.wav"
audio_have = "/home/ares/hgj/NLP/ass2/speech_recognition/examples/microphone-results.wav"
model_path = "home/ares/anaconda3/envs/nlp/lib/python3.10/site-packages/pyAudioAnalysis/data/models/svm_rbf_4class"  # 修改为实际模型路径
model_type = "svm_rbf"
emotion_classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

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

# 使用pyAudioAnalysis进行情绪识别
def classify_emotion(audio_file, model_path, model_type):
    [flags_ind, classes_all, acc, CM] = aS.mid_term_file_classification(audio_file, model_path, model_type)

    # 输出分类结果
    for i, flag in enumerate(flags_ind):
        # 将flag转换为整数
        int_flag = int(flag)
        print(f"Segment {i}: {emotion_classes[int_flag]}")

    print(f"Classes: {classes_all}")
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix: {CM}")

def main():
    choice = input("请输入1进行录音，或输入2读取音频文件: ")

    if choice == '1':
        record_audio(audio_file)
        classify_emotion(audio_file, model_path, model_type)
    elif choice == '2':
        # audio_have = input("请输入音频文件路径: ")
        print("输入的音频文件是:" + audio_have)
        classify_emotion(audio_have, model_path, model_type)
    else:
        print("无效选择，请输入1或2")

if __name__ == "__main__":
    main()
