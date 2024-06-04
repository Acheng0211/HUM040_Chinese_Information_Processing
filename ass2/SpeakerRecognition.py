from __future__ import print_function
import os
import csv
import glob
import scipy
import sklearn
import numpy as np
import hmmlearn.hmm
import sklearn.cluster
import pickle as cpickle
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sklearn.discriminant_analysis
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))
import pyAudioAnalysis.audioBasicIO as audioBasicIO
import pyAudioAnalysis.audioTrainTest as at
import pyAudioAnalysis.MidTermFeatures as mtf
import pyAudioAnalysis.ShortTermFeatures as stf
import os
import speech_recognition as sr
from pyAudioAnalysis import audioSegmentation as aS

current_dir = os.path.dirname(__file__)
# 定义录音保存路径
audio_file = "microphone-results.wav"
audio_have = current_dir + "/" +  "speech_recognition/examples/microphone-results.wav"
speaker_file = current_dir + "/" + "pyAudioAnalysis/pyAudioAnalysis/data/diarizationExample.wav"
sm_file = current_dir + "/" + "pyAudioAnalysis/pyAudioAnalysis/data/scottish.wav"
model_path = current_dir + "/" + "pyAudioAnalysis/pyAudioAnalysis/data/models/svm_rbf_sm" 
model_path_4class = current_dir + "/" + "pyAudioAnalysis/pyAudioAnalysis/data/models/svm_rbf_4class" 
model_type = "svm_rbf"
numSpeakers = 4
# 定义情绪类别
emotion_classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def speaker_diarization(filename, n_speakers, mid_window=1.0, mid_step=0.1,
                        short_window=0.1, lda_dim=0, plot_res=False):
    """
    ARGUMENTS:
        - filename:        the name of the WAV file to be analyzed
        - n_speakers       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - mid_window (opt)    mid-term window size
        - mid_step (opt)    mid-term window step
        - short_window  (opt)    short-term window size
        - lda_dim (opt     LDA dimension (0 for no LDA)
        - plot_res         (opt)   0 for not plotting the results 1 for plotting
    """
    sampling_rate, signal = audioBasicIO.read_audio_file(filename)
    signal = audioBasicIO.stereo_to_mono(signal)
    duration = len(signal) / sampling_rate

    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "data/models")

    classifier_all, mean_all, std_all, class_names_all, _, _, _, _, _ = \
        at.load_model(os.path.join(base_dir, "svm_rbf_speaker_10"))
    classifier_fm, mean_fm, std_fm, class_names_fm, _, _, _, _,  _ = \
        at.load_model(os.path.join(base_dir, "svm_rbf_speaker_male_female"))


    mid_feats, st_feats, a = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mid_window * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * 0.05),
                                   round(sampling_rate * 0.05))

    mid_term_features = np.zeros((mid_feats.shape[0] + len(class_names_all) +
                                  len(class_names_fm), mid_feats.shape[1]))
    for index in range(mid_feats.shape[1]):
        feature_norm_all = (mid_feats[:, index] - mean_all) / std_all
        feature_norm_fm = (mid_feats[:, index] - mean_fm) / std_fm
        _, p1 = at.classifier_wrapper(classifier_all, "svm_rbf", feature_norm_all)
        _, p2 = at.classifier_wrapper(classifier_fm, "svm_rbf", feature_norm_fm)
        start = mid_feats.shape[0]
        end = mid_feats.shape[0] + len(class_names_all)
        mid_term_features[0:mid_feats.shape[0], index] = mid_feats[:, index]
        mid_term_features[start:end, index] = p1 + 1e-4
        mid_term_features[end::, index] = p2 + 1e-4
    # normalize features:
    scaler = StandardScaler()
    mid_feats_norm = scaler.fit_transform(mid_term_features.T)

    # remove outliers:
    dist_all = np.sum(distance.squareform(distance.pdist(mid_feats_norm.T)),
                      axis=0)
    m_dist_all = np.mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.1 * m_dist_all)[0]

    # TODO: Combine energy threshold for outlier removal:
    # EnergyMin = np.min(mt_feats[1,:])
    # EnergyMean = np.mean(mt_feats[1,:])
    # Thres = (1.5*EnergyMin + 0.5*EnergyMean) / 2.0
    # i_non_outliers = np.nonzero(mt_feats[1,:] > Thres)[0]
    # print i_non_outliers

    mt_feats_norm_or = mid_feats_norm
    mid_feats_norm = mid_feats_norm[:, i_non_outliers]

    # LDA dimensionality reduction:
    if lda_dim > 0:

        # extract mid-term features with minimum step:
        window_ratio = int(round(mid_window / short_window))
        step_ratio = int(round(short_window / short_window))
        mt_feats_to_red = []
        num_of_features = len(st_feats)
        num_of_stats = 2
        for index in range(num_of_stats * num_of_features):
            mt_feats_to_red.append([])

        # for each of the short-term features:
        for index in range(num_of_features):
            cur_pos = 0
            feat_len = len(st_feats[index])
            while cur_pos < feat_len:
                n1 = cur_pos
                n2 = cur_pos + window_ratio
                if n2 > feat_len:
                    n2 = feat_len
                short_features = st_feats[index][n1:n2]
                mt_feats_to_red[index].append(np.mean(short_features))
                mt_feats_to_red[index + num_of_features].\
                    append(np.std(short_features))
                cur_pos += step_ratio
        mt_feats_to_red = np.array(mt_feats_to_red)
        mt_feats_to_red_2 = np.zeros((mt_feats_to_red.shape[0] +
                                      len(class_names_all) +
                                      len(class_names_fm),
                                      mt_feats_to_red.shape[1]))
        limit = mt_feats_to_red.shape[0] + len(class_names_all)
        for index in range(mt_feats_to_red.shape[1]):
            feature_norm_all = (mt_feats_to_red[:, index] - mean_all) / std_all
            feature_norm_fm = (mt_feats_to_red[:, index] - mean_fm) / std_fm
            _, p1 = at.classifier_wrapper(classifier_all, "svm_rbf",
                                          feature_norm_all)
            _, p2 = at.classifier_wrapper(classifier_fm, "svm_rbf", feature_norm_fm)
            mt_feats_to_red_2[0:mt_feats_to_red.shape[0], index] = \
                mt_feats_to_red[:, index]
            mt_feats_to_red_2[mt_feats_to_red.shape[0]:limit, index] = p1 + 1e-4
            mt_feats_to_red_2[limit::, index] = p2 + 1e-4
        mt_feats_to_red = mt_feats_to_red_2
        scaler = StandardScaler()
        mt_feats_to_red = scaler.fit_transform(mt_feats_to_red.T).T
        labels = np.zeros((mt_feats_to_red.shape[1], ))
        lda_step = 1.0
        lda_step_ratio = lda_step / short_window
        for index in range(labels.shape[0]):
            labels[index] = int(index * short_window / lda_step_ratio)
        clf = sklearn.discriminant_analysis.\
            LinearDiscriminantAnalysis(n_components=lda_dim)
        mid_feats_norm = clf.fit_transform(mt_feats_to_red.T, labels)
        #clf.fit(mt_feats_to_red.T, labels)
        #mid_feats_norm = (clf.transform(mid_feats_norm.T)).T
    if n_speakers <= 0:
        s_range = range(2, 10)
    else:
        s_range = [n_speakers]
    cluster_labels = []
    sil_all = []
    cluster_centers = []
    
    for speakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=speakers)
        k_means.fit(mid_feats_norm)
        cls = k_means.labels_ 
        cluster_labels.append(cls)
#        cluster_centers.append(means)
        sil_1 = []; sil_2 = []
        for c in range(speakers):
            # for each speaker (i.e. for each extracted cluster)
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if clust_per_cent < 0.020:
                sil_1.append(0.0)
                sil_2.append(0.0)
            else:
                # get subset of feature vectors
                mt_feats_norm_temp = mid_feats_norm[cls == c, :]
                # compute average distance between samples
                # that belong to the cluster (a values)
                dist = distance.pdist(mt_feats_norm_temp.T)
                sil_1.append(np.mean(dist)*clust_per_cent)
                sil_temp = []
                for c2 in range(speakers):
                    # compute distances from samples of other clusters
                    if c2 != c:
                        clust_per_cent_2 = np.nonzero(cls == c2)[0].shape[0] /\
                                           float(len(cls))
                        mid_features_temp = mid_feats_norm[cls == c2, :]
                        dist = distance.cdist(mt_feats_norm_temp,
                                              mid_features_temp)
                        sil_temp.append(np.mean(dist)*(clust_per_cent
                                                       + clust_per_cent_2)/2.0)
                sil_temp = np.array(sil_temp)
                # ... and keep the minimum value (i.e.
                # the distance from the "nearest" cluster)
                sil_2.append(min(sil_temp))
        sil_1 = np.array(sil_1)
        sil_2 = np.array(sil_2)
        sil = []
        for c in range(speakers):
            # for each cluster (speaker) compute silhouette
            sil.append((sil_2[c] - sil_1[c]) / (max(sil_2[c], sil_1[c]) + 1e-5))
        # keep the AVERAGE SILLOUETTE
        sil_all.append(np.mean(sil))

    imax = int(np.argmax(sil_all))
    # optimal number of clusters
    num_speakers = s_range[imax]

    # generate the final set of cluster labels
    # (important: need to retrieve the outlier windows:
    # this is achieved by giving them the value of their
    # nearest non-outlier window)
#    print(cls)
#    cls = np.zeros((n_wins,))
#    for index in range(n_wins):
#        j = np.argmin(np.abs(index-i_non_outliers))
#        cls[index] = cluster_labels[imax][j]
    # Post-process method 1: hmm smoothing
    if lda_dim <= 0 :
        for index in range(1):
            # hmm training
            start_prob, transmat, means, cov = \
                train_hmm_compute_statistics(mt_feats_norm_or.T, cls)
            hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], "diag")
            hmm.startprob_ = start_prob
            hmm.transmat_ = transmat            
            hmm.means_ = means; hmm.covars_ = cov
            cls = hmm.predict(mt_feats_norm_or)                        
    # Post-process method 2: median filtering:
    cls = scipy.signal.medfilt(cls, 5)

    class_names = ["speaker{0:d}".format(c) for c in range(num_speakers)]

    # load ground-truth if available
    gt_file = filename.replace('.wav', '.segments')
    # if groundtruth exists
    if os.path.isfile(gt_file):
        seg_start, seg_end, seg_labs = read_segmentation_gt(gt_file)
        flags_gt, class_names_gt = segments_to_labels(seg_start, seg_end,
                                                      seg_labs, mid_step)

    if plot_res:
        fig = plt.figure()    
        if n_speakers > 0:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(np.array(range(len(cls))) * mid_step + mid_step / 2.0, cls)

    purity_cluster_m, purity_speaker_m = -1, -1
    if os.path.isfile(gt_file):
        if plot_res:
            ax1.plot(np.array(range(len(flags_gt))) *
                     mid_step + mid_step / 2.0, flags_gt, 'r')
        purity_cluster_m, purity_speaker_m = \
            evaluate_speaker_diarization(cls, flags_gt)
        print("{0:.1f}\t{1:.1f}".format(100 * purity_cluster_m,
                                        100 * purity_speaker_m))
        if plot_res:
            plt.title("Cluster purity: {0:.1f}% - "
                      "Speaker purity: {1:.1f}%".format(100 * purity_cluster_m,
                                                        100 * purity_speaker_m))
    if plot_res:
        plt.xlabel("time (seconds)")
        if n_speakers <= 0:
            plt.subplot(212)
            plt.plot(s_range, sil_all)
            plt.xlabel("number of clusters")
            plt.ylabel("average clustering's sillouette")
        plt.show()
    return cls, purity_cluster_m, purity_speaker_m


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

def classify_SM(audio_file, model_path, model_type):
    if not os.path.isfile(model_path):
        raise Exception("Input model_name not found!")
    gtFile = ""
    if audio_file[-4::]==".wav":
        gtFile = audio_file.replace(".wav", ".segments")
    
    aS.mid_term_file_classification(audio_file, model_path, model_type, True, gtFile)

def main():
    choice = input('请输入"1"识别发音人，输入"2"识别语音类型，输入"3"识别情绪: ')

    if choice == '1':
        aS.speaker_diarization(speaker_file, numSpeakers, lda_dim=0, plot_res=True)
    elif choice == '2':
        classify_SM(sm_file, model_path, model_type)    
    elif choice == '3':
        choice1 = input("请输入4进行录音，或输入5读取音频文件: ")
        if choice1 == '4':
            record_audio(audio_file)
            classify_emotion(audio_file, model_path_4class, model_type)
        elif choice1 == '5':
            # audio_have = input("请输入音频文件路径: ")
            print("输入的音频文件是:" + audio_have)
            classify_emotion(audio_have, model_path_4class, model_type)
        else:
            print("无效选择，请输入4或5")
    else:
        print("无效选择，请输入1、2或3")

if __name__ == "__main__":
    main()
