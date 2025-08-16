import pandas as pd
import pickle

# 加载数据
print("正在加载数据...")

# 加载物品ID到描述的映射
df_dict = pd.read_csv("../data/combined_reads_asin_mapping.csv")
title_id_dict, descript_id_dict = {}, {}  # 初始化映射字典
for i in range(len(df_dict)):
    descript_id_dict[df_dict['index'].iloc[i]] = df_dict['description'].iloc[i]
    title_id_dict[df_dict['index'].iloc[i]] = df_dict['title'].iloc[i]

# 从descript_id_dict中获取空描述的index
print("正在查找空描述的index...")
null_descript_dict = {}
for index, description in descript_id_dict.items():
    # 检查描述是否为空（None、空字符串、或只包含空白字符）
    # if description == "['', '']" or str(description) == '[Nan]' or str(description) == "['', '', '', '']" or str(description) == "['', '', '', '', '', '']":
    null_descript_dict[index] = [title_id_dict[index], [], str(description)]  # 添加空的description字段

print(f"找到 {len(null_descript_dict)} 个空描述的index")

# 加载yelpAll_recagent.pkl
print("正在加载yelpAll_recagent.pkl...")
yelp_data = pd.read_pickle("../data/readsAll_recagent.pkl")

print(f"yelpAll_recagent.pkl包含 {len(yelp_data)} 条记录")

# 遍历yelpAll_recagent.pkl
print("正在遍历数据并收集评论...")
for i, record in enumerate(yelp_data.iterrows()):
    if i % 1000 == 0:
        print(f"已处理 {i} 条记录")
    
    # 获取当前记录的asin列表和评论
    asin_list = record[1]['asin']  # record[1]是Series数据
    review_text = record[1]['reviewText']
    
    # 检查asin列表中的每个元素是否存在于null_descript_dict中
    for j, asin_id in enumerate(asin_list):
        if asin_id in null_descript_dict:
            # 将对应的评论加入到null_descript_dict中
            if j < len(review_text):  # 确保索引不越界
                null_descript_dict[asin_id][1].append(review_text[j])

print("处理完成！")

# 保存结果
print("\n正在保存结果到文件...")

# 将null_descript_dict转换为DataFrame格式
result_data = []
for index, (title, reviews, description) in null_descript_dict.items():
    # 删除评论中的回车符，并在每条评论前添加序号
    if reviews:
        numbered_reviews = []
        for i, review in enumerate(reviews, 1):
            # 删除回车符和换行符
            clean_review = str(review).replace('\n', ' ').replace('\r', ' ')
            numbered_reviews.append(f"{i}. {clean_review}")
        reviews_str = '\n'.join(numbered_reviews)
    else:
        reviews_str = ''
    
    result_data.append({
        'index': str(index),
        'title': title,
        'reviews': reviews_str,
        'review_count': len(reviews),
        'description': description  # 添加description字段，当前为空
    })

# 创建DataFrame并保存为CSV
result_df = pd.DataFrame(result_data)
result_df.to_csv('collect_reads_description.csv', index=False, encoding='utf-8-sig')
print("结果已保存到 collect_description.csv")


print(f"总共处理了 {len(null_descript_dict)} 个空描述的index")
total_reviews = sum(len(reviews) for _, reviews, _ in null_descript_dict.values())
print(f"总共收集了 {total_reviews} 条评论")