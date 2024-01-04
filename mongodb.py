from pymongo import MongoClient
from datetime import datetime

# 日志文件路径
log_file_path = "/Users/zhuidexiaopengyou/Downloads/movie_recommend/rating_log.txt"  # 更改为你的日志文件路径

# 连接到MongoDB
client = MongoClient("mongodb://localhost:27017/")

# 创建或连接到数据库"user_behavior_db"
db = client["user_behavior_db"]

# 创建或连接到集合"ratings"
ratings = db["ratings"]

# 读取日志文件并将数据插入或更新到"ratings"集合中
with open(log_file_path, "r") as file:
    for line in file:
        try:
            username, movie_id, title, rating, timestamp = line.strip().split(", ")
            rating = int(rating)  # 确保评分是整数
            timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")  # 解析时间戳

            # 查询用户是否已经存在于数据库中
            user_record = db.users.find_one({"username": username})
            if not user_record:
                # 如果用户不存在，就添加用户
                user_id = db.users.insert_one({
                    "username": username,
                    "registration_date": datetime.now()
                }).inserted_id
            else:
                user_id = username

            # 检查是否存在相同的评分记录
            existing_rating = ratings.find_one({"user_id": user_id, "movie_id": movie_id})
            if existing_rating:
                if existing_rating['rating'] != rating:  # 如果评分不一致，则更新记录
                    ratings.update_one(
                        {"_id": existing_rating["_id"]},
                        {"$set": {
                            "rating": rating,
                            "rating_time": timestamp  # 更新时间戳为最新的评分时间
                        }}
                    )
            else:
                # 插入新的评分记录
                ratings.insert_one({
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "title": title,
                    "rating": rating,
                    "rating_time": timestamp
                })

        except ValueError as ve:
            print(f"数据格式错误: {line}. 错误: {ve}")
        except Exception as e:
            print(f"处理 {line} 时发生错误: {e}")

print("日志数据已导入或更新到数据库中。")
