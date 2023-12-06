from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
# 1. 加载电影数据集
data = pd.read_csv('/Users/zhuidexiaopengyou/Downloads/movie_recommend/netflix_titles.csv')  # 替换为你的CSV文件路径

# 2. 数据预处理（如果需要的话）
# 在这一步中，你可以进行数据清理和转换，确保数据格式正确。如果数据已经包含"title"和"description"列，就无需额外的处理。

# 3. 合并属性到一个文本
data['combined_features'] = data.apply(lambda row: ' '.join([str(row['type']),
                                                            str(row['title']),
                                                            str(row['director']),
                                                            str(row['cast']),
                                                            str(row['country']),
                                                            str(row['date_added']),
                                                            str(row['release_year']),
                                                            str(row['rating']),
                                                            str(row['duration']),
                                                            str(row['listed_in']),
                                                            str(row['description'])]), axis=1)
# 首先，对类型和描述进行TF-IDF向量化，以便计算相似度
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_genres = tfidf_vectorizer.fit_transform(data['listed_in'].fillna(''))
tfidf_matrix_description = tfidf_vectorizer.fit_transform(data['description'].fillna(''))
# 分割演员字符串和导演字符串为列表，确保非空值后进行分割
data['cast_list'] = data['cast'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
data['director_list'] = data['director'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

def enhanced_overlap(title, data, top_n=5):
    movie_data = data[data['title'] == title].iloc[0]
    directors = set(movie_data['director_list'])
    cast = set(movie_data['cast_list'])
    genres = movie_data['listed_in']
    description = movie_data['description']
    
    def calculate_normalized_overlap(row):
        # 加权导演得分
        director_score = 2 * len(directors.intersection(set(row['director_list']))) / 2 # 最高可能得分为2

        # 计算共享演员的比例得分
        cast_score = (len(cast.intersection(set(row['cast_list']))) / len(row['cast_list'])) if row['cast_list'] else 0

        # 计算类型相似度
        genres_similarity = cosine_similarity(tfidf_matrix_genres[movie_data.name].reshape(1, -1),
                                            tfidf_matrix_genres[row.name].reshape(1, -1))[0][0]

        # 计算描述文本相似度
        description_similarity = cosine_similarity(tfidf_matrix_description[movie_data.name].reshape(1, -1),
                                                tfidf_matrix_description[row.name].reshape(1, -1))[0][0]

        # 综合得分
        total_score = director_score + cast_score + genres_similarity + description_similarity
        normalized_score = total_score / 4 # 最高可能得分为4

        return normalized_score

    # 应用新的计算函数
    data['overlap_score'] = data.apply(calculate_normalized_overlap, axis=1)


    # 计算所有电影的综合得分
    data['overlap_score'] = data.apply(calculate_normalized_overlap, axis=1)
    
    # 返回得分最高的top_n个电影
    recommendations = data[data['title'] != title].sort_values('overlap_score', ascending=False).head(top_n)
    return recommendations[['title', 'overlap_score']]

# 获取增强型规则推荐结果
enhanced_recommendations = enhanced_overlap('Blood & Water', data)
print(enhanced_recommendations)
