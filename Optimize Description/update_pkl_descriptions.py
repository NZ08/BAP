import pandas as pd
import pickle

# 读取description字典
print("正在读取description字典...")
description_df = pd.read_csv('collect_reads_description_with_generated.csv')
description_dict = description_df.set_index('index')['description'].to_dict()
print(f"读取到 {len(description_dict)} 条description记录")

# 读取pkl文件
print("正在读取pkl文件...")
with open('../data/readsAll_recagent.pkl', 'rb') as f:
    books_data = pickle.load(f)

print(f"pkl文件包含 {len(books_data)} 条记录")
print(f"数据列名: {books_data.columns.tolist()}")

# 统计更新数量
updated_count = 0

# 重置索引
books_data = books_data.reset_index(drop=True)

# 遍历每一行，更新description
print("开始更新description...")
for idx, row in books_data.iterrows():
    asin_list = row['asin']  # 获取asin列的值（应该是一个列表）
    description_list = row['description']  # 获取description列的值
        
    # 遍历asin列表中的每个值
    for i, asin in enumerate(asin_list):
        # 如果asin在description_dict中，直接替换对应位置的description
        if asin in description_dict and description_dict[asin] != '生成失败':
            description_list[i] = str([description_dict[asin]])
            updated_count += 1
    
    # 更新回原数据
    books_data.at[idx, 'description'] = description_list

    # idx是索引，count是计数
    # 每处理1000行显示进度
    if (idx + 1) % 1000 == 0:
        print(f"已处理 {idx + 1} 行，更新了 {updated_count} 个description")

print(f"\n处理完成！")
print(f"成功更新了 {updated_count} 个description")

# 保存更新后的pkl文件
output_file = 'readsAll_recagent_updated.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(books_data, f)

print(f"更新后的数据已保存到: {output_file}")