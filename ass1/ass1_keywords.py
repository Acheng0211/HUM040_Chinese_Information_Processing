import jieba
from snownlp import SnowNLP
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# from textblob import TextBlob


def jieba_test():
    text = "今天天气很好，我去公园玩了。"

    #精确模式,试图将句子最精确地切开，适合文本分析
    seg = jieba.lcut(text, cut_all  = False) 
    print(' '+ "/ ".join(seg))
    #全模式，把句子中所有的可以成词的词语都扫描出来，但是不能解决歧义
    seg = jieba.lcut(text, cut_all  = True) 
    print(' '+ "/ ".join(seg))
    #搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细。
    seg =jieba.lcut_for_search(text)
    print(' '+ "/ ".join(seg))

def Emotion_analysis(text_emo):
    # 去除文本中的标点符号
    text2 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", text_emo)
    # 使用jieba进行分词
    seg_list = jieba.lcut(text2, cut_all=False)

    # # 创建TextBlobCN对象
    # blob = TextBlob(text)
    # # 进行情感分析
    # TB_sentiment_score = blob.sentiment.polarity
    # # 判断情感极性
    # TB_sentiment = "正面" if TB_sentiment_score > 0 else "负面" if TB_sentiment_score < 0 else "中性"
    # print("TextBlobCN文本情感分析结果：", TB_sentiment, "，情感分数：", TB_sentiment_score)

    # 使用SnowNLP进行情感分析
    s = SnowNLP(text_emo)
    sentiment_score = s.sentiments
    # 判断情感极性
    sentiment = "正面" if sentiment_score > 0.66 else "负面"
    print("分词结果：", seg_list)
    print("SnowNLP文本情感分析结果：", sentiment, "，情感分数：", sentiment_score)

def Extract_keywords(text_raw):
    # 去除文本中的标点符号
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", text_raw)

    # 使用jieba进行分词
    seg_list = jieba.lcut(text, cut_all=False)

    # 使用SnowNLP进行关键词提取
    s = SnowNLP(text)
    keywords_snownlp = s.keywords(limit=10)

    # 使用TF-IDF算法提取关键词
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(seg_list)])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # 提取TF-IDF值最高的10个关键词
    top_keywords_index = tfidf_matrix.toarray().argsort()[0][-10:][::-1]
    keywords_tfidf = [feature_names[index] for index in top_keywords_index]

    print("分词结果：", seg_list)
    print("SnowNLP关键词提取结果：", keywords_snownlp)
    print("TF-IDF关键词提取结果：", keywords_tfidf)

    # 计算正确率、召回率和F值
    #reference_keywords = set(["医学", "人工智能", "机器学习", "大数据", "分析", "技术", "疾病", "数据", "隐私", "保护", "算法", "透明度", "研发", "快速", "准确"])
    reference_keywords = set(["医学", "人工智能","人工","智能", "机器学习", "大数据", "疾病"])
    extracted_keywords_snownlp = set(keywords_snownlp)
    extracted_keywords_tfidf = set(keywords_tfidf)
    # 计算正确率
    correct_keywords = [extracted_keywords_snownlp.intersection(reference_keywords), extracted_keywords_tfidf.intersection(reference_keywords)]
    precision = [len(correct_keywords[0]) / len(extracted_keywords_snownlp) if len(extracted_keywords_snownlp) > 0 else 0, len(correct_keywords[1]) / len(extracted_keywords_tfidf) if len(extracted_keywords_tfidf) > 0 else 0]
    # 计算召回率
    recall = [len(correct_keywords[0]) / len(reference_keywords) if len(reference_keywords) > 0 else 0, len(correct_keywords[1]) / len(reference_keywords) if len(reference_keywords) > 0 else 0]
    # 计算F值
    f_score = [2 * precision[0] * recall[0] / (precision[0] + recall[0]) if precision[0] + recall[0] > 0 else 0, 2 * precision[1] * recall[1] / (precision[1] + recall[1]) if precision[1] + recall[1] > 0 else 0]

    print("正确率（Precision）：", precision[0], "(SnowNLP)," ,precision[1],"(TF-IDF)")
    print("召回率（Recall）：", recall[0],"(SnowNLP),",recall[1],"(TF-IDF)")
    print("F值（F-score）：", f_score[0],"(SnowNLP),",f_score[1],"(TF-IDF)")


if __name__ == "__main__":
    # 中文文本
    text = "南方科技大学，简称“南科大”，位于广东省深圳市，是国家“双一流”建设高校、国家高等教育综合改革试验校、广东省高水平理工科大学、广东省高水平大学，入选深圳国际友好城市大学联盟、深圳高校创新创业教育联盟，是深圳市在中国高等教育改革发展的时代背景下、主要借鉴香港科技大学办学经验创建的一所公办新型研究型大学。"
    text_raw = "题目：人工智能在医学领域的应用及前景分析。人工智能在医学领域的应用越来越受到关注。通过机器学习和大数据分析，人工智能技术可以帮助医生快速准确地诊断疾病，例如通过分析医疗影像图像来发现异常。未来，随着技术的进步，人工智能有望在疾病预防、治疗方案设计等方面发挥更重要的作用。然而，人工智能在医学领域的应用也面临着一些挑战，例如医疗数据的隐私保护和算法的透明度。因此，需要进一步加强技术研发和跨界合作，推动人工智能在医学领域的健康发展。"
    text_emo_positve = "南方科技大学是一所非常优秀的大学，拥有优秀的师资队伍和先进的教学设施，培养了许多优秀的人才，为社会做出了重要贡献。"
    text_emo_negative = "南方科技大学真差、环境糟糕、饭堂不好吃"
    text_emo_negative_complex = "南方科技大学的教学质量很差，师资力量不足，教学设施陈旧，学生毕业后很难找到工作。"

    Extract_keywords(text_raw)
    #Emotion_analysis(text_emo_positve)
    #Emotion_analysis(text_emo_negative)
