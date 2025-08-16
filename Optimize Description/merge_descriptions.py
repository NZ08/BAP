import pandas as pd

# 读取两个CSV文件
print("正在读取文件...")

# 读取包含description的文件
description_df = pd.read_csv('collect_reads_description_with_generated.csv')
print(f"读取到 {len(description_df)} 条description记录")

# 读取目标文件
target_df = pd.read_csv('../data/combined_reads_asin_mapping.csv')
print(f"读取到 {len(target_df)} 条目标记录")

# 将description_df的index设为索引，方便查找
description_dict = description_df.set_index('index')['description'].to_dict()

# 创建新的description列
new_descriptions = []
matched_count = 0

for idx, row in target_df.iterrows():
    index_value = row['index']
    if index_value in description_dict and description_dict[index_value] != '生成失败':
        new_descriptions.append(str([description_dict[index_value]]))
        matched_count += 1
    else:
        # 如果没有匹配的description，保持原有的description
        new_descriptions.append(row['description'])

# 更新target_df的description列
target_df['description'] = new_descriptions

print(f"成功匹配并更新了 {matched_count} 条记录")

# 保存到新文件
output_file = 'combined_reads_with_descriptions.csv'
target_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"结果已保存到: {output_file}")

print("\n处理完成！")
print(f"输出文件包含 {len(target_df)} 条记录")
print(f"其中 {matched_count} 条记录的description已更新")