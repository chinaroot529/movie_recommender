from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
data = pd.read_csv('/Users/zhuidexiaopengyou/Downloads/movie_recommend/netflix_titles.csv')  # 替换为你的CSV文件路径
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
# 使用CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count_vectorizer.fit_transform(data['combined_features'])

# 计算余弦相似度
cosine_sim_count = cosine_similarity(count_matrix, count_matrix)

# 处理NaN值，将其替换为None
data = data.where(pd.notna(data), None)

print(cosine_sim_count)