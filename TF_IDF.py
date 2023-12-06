from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
CORS(app)

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

# 4. 使用TF-IDF向量化合并后的文本
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'].fillna(''))  # 填充缺失的文本字段

# 5. 计算电影之间的相似性
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 处理NaN值，将其替换为None
data = data.where(pd.notna(data), None)

# 6. 定义一个函数来获取电影推荐
def get_recommendations(movie_title, cosine_sim=cosine_sim, top_n=10):
    # 检查用户输入的电影名称是否在数据集中
    if not data['title'].eq(movie_title).any():
        return jsonify({'error': '电影名称不存在'})

    # 获取电影的索引
    idx = data[data['title'] == movie_title].index[0]

    # 计算电影与其他电影的相似性得分
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 根据相似性得分排序电影
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 获取前top_n个相似电影的索引和相似度得分
    sim_scores = sim_scores[1:(top_n+1)]
    movie_indices = [i[0] for i in sim_scores]
    sim_scores = [i[1] for i in sim_scores]

    # 获取这些电影的标题和描述
    recommended_movies = data['title'].iloc[movie_indices]
    movie_descriptions = data['description'].iloc[movie_indices]

    return recommended_movies, movie_descriptions, sim_scores

@app.route('/api/get_recommendations', methods=['GET'])
def recommend_movies():
    movie_title = request.args.get('movieTitle')

    if not movie_title:
        return jsonify({'error': '请输入电影名称'})

    # 检查电影名称是否在数据集中
    if movie_title not in data['title'].values.tolist():
        return jsonify({'error': '电影名称不存在'})

    recommendations, movie_descriptions, similarity_scores = get_recommendations(movie_title)

    if recommendations.empty:
        return jsonify({'message': '没有找到相关的推荐电影'})

    # 只返回用于计算相似度和推荐的数据
    response_data = {
        'recommendations': recommendations.tolist(),
        'similarity_scores': [round(score, 4) for score in similarity_scores],
        'movie_descriptions': movie_descriptions.tolist()
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
