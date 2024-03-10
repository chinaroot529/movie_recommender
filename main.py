from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
from pymongo import MongoClient
import numpy as np
from bson import ObjectId  # 这是MongoDB的库

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# 连接到MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["user_behavior_db"]
movie_dataset = db["dataset"]  # 替换为你的电影数据集的实际名称


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
# 使用TF-IDF矩阵计算物品相似度
item_similarity = cosine_similarity(tfidf_matrix)

# 将相似度矩阵转换为DataFrame，方便后续处理
item_similarity_df = pd.DataFrame(item_similarity, index=data['title'], columns=data['title'])

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

# 7.获取个性化推荐内容

def get_user_ratings(user_id):
    """获取用户的评分记录"""
    # 确保从ratings集合中获取的是对应用户的评分记录
    return list(db.ratings.find({"user_id": user_id}))

def get_all_users_ratings():
    """获取所有用户的评分记录"""
    return list(db.ratings.find())

def calculate_similarity(target_ratings, all_ratings,user_id):
    """计算目标用户与所有用户的相似度"""
    similarities = {}
    for user_rating in all_ratings:
        if user_rating['user_id'] == user_id:  # 跳过目标用户自己
            continue
        user_vector = [r['rating'] for r in all_ratings if r['user_id'] == user_rating['user_id']]
        if not user_vector or not target_ratings:  # 如果用户没有评分或目标用户没有评分，则跳过
            continue
        # 确保向量长度一致
        min_len = min(len(target_ratings), len(user_vector))
        target_vector = np.array([r['rating'] for r in target_ratings][:min_len])
        user_vector = np.array(user_vector[:min_len])
        similarity = cosine_similarity([target_vector], [user_vector])
        similarities[user_rating['user_id']] = similarity[0][0]
    return similarities

def get_personal_recommendations(target_user_id, top_n=20):
    """生成推荐"""
    target_ratings = get_user_ratings(target_user_id)
    all_ratings = get_all_users_ratings()
    similarities = calculate_similarity(target_ratings, all_ratings,target_user_id)

    similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = []
    for user_id, _ in similar_users:
        user_ratings = get_user_ratings(user_id)
        for rating in user_ratings:
            if rating['movie_id'] not in [r['movie_id'] for r in target_ratings] and \
                    movie_dataset.find_one({"show_id": rating['movie_id']}) is not None:  # 确保movie_id有效
                recommendations.append(rating['movie_id'])

    return recommendations[:top_n]

def get_item_based_recommendations(user_id, top_n=10):
    # 获取用户评分过的电影
    user_ratings = get_user_ratings(user_id)  # 使用你之前的函数获取用户评分
    user_ratings = pd.DataFrame(user_ratings)
    user_ratings = user_ratings[user_ratings['user_id'] == user_id]

    # 从用户评分高的电影开始推荐
    user_ratings = user_ratings.sort_values(by='rating', ascending=False)

    recommendations = {}
    for _, row in user_ratings.iterrows():
        movie_title = row['movie_id']  # 确保这是电影标题
        similar_movies = item_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+1]
        for similar_movie, score in similar_movies.items():
            if similar_movie not in recommendations:
                recommendations[similar_movie] = score

    # 根据得分排序推荐结果
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

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

@app.route('/api/static_recommendations', methods=['GET'])
def static_recommendations():
    top_n = 25  # number of movies per page

    # Randomly select movies to simulate a dynamic 'Top' list
    recommendations = data.sample(n=top_n)

    # Construct response data
    response_data = {
        'movies': []
    }
    for _, row in recommendations.iterrows():
        response_data['movies'].append({
            'rank': _ + 1,  # Add ranking
            'title': row['title'],
            'director': row['director'],
            'cast': row['cast'],
            'release_year': row['release_year'],
            'country': row['country'],
            'listed_in': row['listed_in'],
            'description': row['description'],
            'rating':row['rating']
        })

    return jsonify(response_data)

users = {'admin': 'password123','xjh':'123'}
@app.route('/api/login', methods=['POST'])
def login():
    try:
        username = request.json.get('username')
        password = request.json.get('password')
        
        if username not in users or users[username] != password:
            return jsonify({'success': False, 'message': '用户名或密码错误'})
        return jsonify({'success': True, 'message': '登录成功'})
    except Exception as e:
        # 适当记录异常 e
        return jsonify({'success': False, 'message': '服务器错误'})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/record_click', methods=['POST'])
def record_click():
    data = request.json  # 获取点击数据
    log_message = f"{data['userId']}, {data['movieId']}, {data['clickTime']}\n"
    
    # 将日志消息写入日志文件
    with open("/Users/zhuidexiaopengyou/Downloads/movie_recommend/click_log.txt", "a") as file:
        file.write(log_message)
    
    return jsonify({'status': 'success'})

@app.route('/api/movie_details', methods=['GET'])
def movie_details():
    # 获取查询参数中的电影名称
    movie_title = request.args.get('title')
    
    # 如果没有提供电影名称或电影名称不在数据集中，则返回错误
    if not movie_title or not data['title'].eq(movie_title).any():
        return jsonify({'error': '电影名称未提供或不存在'})

    # 获取电影的详细信息
    movie_data = data[data['title'] == movie_title].iloc[0].to_dict()

    # 返回电影的详细信息
    return jsonify(movie_data)

@app.route('/api/rate_movie', methods=['POST'])
def rate_movie():
    data = request.json  # Assuming JSON payload with title and rating
    user_id = data.get('userId')
    movie_title = data.get('title')
    movie_id = data.get('movieId')  # Make sure you have movieId in your request body
    rating = data.get('rating')
    
    # 当前时间
    now = datetime.now().isoformat()
    
    # 调用函数来保存评分
    save_rating(user_id, movie_id, movie_title, rating, now)
    
    return jsonify({'status': 'success', 'message': f'{movie_title} rated as {rating}'})

def save_rating(user_id, movie_id, title, rating, timestamp):
    # 这个函数处理将评分保存到数据库或文件的逻辑
    with open('/Users/zhuidexiaopengyou/Downloads/movie_recommend/rating_log.txt', "a") as file:
        file.write(f"{user_id}, {movie_id}, {title}, {rating}, {timestamp}\n")
    print(f"Received rating of {rating} for {title} by user {user_id} at {timestamp}")

@app.route('/api/get_rating', methods=['GET'])
def get_rating():
    user_id = request.args.get('userId')
    movie_id = request.args.get('movieId')
    
    if not user_id or not movie_id:
        return jsonify({'error': '需要提供用户ID和电影ID'})

    rating = db.ratings.find_one({'user_id': user_id, 'movie_id': movie_id})

    if rating:
        return jsonify({'rating': rating['rating']})
    else:
        return jsonify({'rating': 0})  # 如果没有找到评分，返回0

@app.route('/api/item_based_recommendations', methods=['GET'])
def item_based_recommendations():
    user_id = request.args.get('userId')

    if not user_id:
        return jsonify({'error': '请提供用户ID'})

    try:
        recommendations = get_item_based_recommendations(user_id)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/get_personalized_recommendations', methods=['GET'])
def get_personalized_recommendations():
    user_id = request.args.get('userId')

    if not user_id:
        return jsonify({'error': '请提供用户ID'})

    try:
        recommendations = get_personal_recommendations(user_id)  # 确保这个函数返回电影ID列表

        # 获取推荐电影的详细信息
        recommended_movies_info = []
        for movie_id in recommendations:
            movie_info = movie_dataset.find_one({"show_id": movie_id})
            if movie_info:
                movie_info['_id'] = str(movie_info['_id'])  # 转换ObjectId为字符串
                recommended_movies_info.append(movie_info)

        return jsonify(recommended_movies_info)  # 返回电影详情的列表

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/find_movie', methods=['GET'])
def find_movie():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': '请提供电影名称'}), 400
    print(title)
    # 修改这里的代码
    movie = next((row.to_dict() for index, row in data.iterrows() if row['title'] == title), None)
    if movie:
        return jsonify(movie)
    else:
        return jsonify({'error': '电影未找到'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)


