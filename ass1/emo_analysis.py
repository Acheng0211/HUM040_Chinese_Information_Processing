import thulac

if __name__ == "__main__":

    # 加载THULAC分词器
    thu = thulac.thulac()

    # 加载情感词典
    sentiment_dict = {}
    with open("hownet_pos.txt", encoding="utf-8") as f:
        for line in f:
            word, sentiment = line.strip().split("\t")
            sentiment_dict[word] = sentiment

    # 中文文本
    # text = "南方科技大学是一所非常优秀的大学，拥有优秀的师资队伍和先进的教学设施，培养了许多优秀的人才，为社会做出了重要贡献。"
    text = "fuck you"


    # 分词
    seg_list = thu.cut(text, text=True).split()

    # 计算文本情感
    positive_words = [word for word in seg_list if word in sentiment_dict and sentiment_dict[word] == "Pos"]
    negative_words = [word for word in seg_list if word in sentiment_dict and sentiment_dict[word] == "Neg"]
    sentiment_score = (len(positive_words) - len(negative_words)) / len(seg_list)

    # 判断情感极性
    sentiment = "正面" if sentiment_score > 0 else "负面" if sentiment_score < 0 else "中性"

    print("文本情感分析结果：", sentiment, "，情感分数：", sentiment_score)
