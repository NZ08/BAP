import pandas as pd
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

GLOBAL_MODEL = "gpt-4o-mini"

# 创建日志文件
log_file = open('generate_reads_descriptions_log.txt', 'w', encoding='utf-8')
log_lock = threading.Lock()  # 添加线程锁保证日志写入安全

def log_print(message):
    """同时打印到控制台和写入文件（线程安全）"""
    with log_lock:
        print(message)
        log_file.write(message + '\n')
        log_file.flush()

# 读取CSV文件
df = pd.read_csv('collect_reads_description.csv')
df['description'] = df['description'].astype('object')

# 初始化OpenAI客户端

client_gpt = OpenAI(
    base_url="https://s.lconai.com/v1/",
    api_key=os.environ.get("LCONAI_API_GPT_KEY"),
)
client_qwen = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)



# 初始化输出CSV文件（创建文件头）
output_file = 'collect_reads_description_with_generated.csv'
if not os.path.exists(output_file):
    # 创建空的结果DataFrame并写入文件头
    result_header = pd.DataFrame(columns=['index', 'description'])
    result_header.to_csv(output_file, index=False, encoding='utf-8-sig')
    log_print(f"已创建输出文件: {output_file}")

def process_single_row(row_data):
    """处理单行数据的函数，用于并行执行"""
    index, asin, title, reviews, description = row_data
    
    # 构建提示词
    prompt = f"Generate an optimized and concise product description (no more than 100 words) based on the following product title, existing description, and user reviews. Please create a new description that removes unnecessary information while keeping the most relevant and useful details:\n\nTitle: {str(title)}\n\nExisting Description: {str(description)[:300]}\n\nUser Reviews: {str(reviews)[:500]}\n\nPlease generate a refined product description in plain text that is more accurate and concise than the existing one. The output should be a complete paragraph without line breaks. Generate description:"
    
    # 定义模型列表，按优先级排序
    models = [
        {"client": client_gpt, "model": "gpt-4o-mini", "name": "gpt-4o-mini"},
        {"client": client_qwen, "model": "qwen-max-latest", "name": "qwen-max-latest"},
        {"client": client_qwen, "model": "qwen-max-2025-01-25", "name": "qwen-max-2025-01-25"},
        {"client": client_qwen, "model": "qwen-max", "name": "qwen-max"},
        {"client": client_qwen, "model": "qwen-plus-latest", "name": "qwen-plus-latest"},
        {"client": client_qwen, "model": "qwen-plus-2025-07-28", "name": "qwen-plus-2025-07-28"},
        {"client": client_qwen, "model": "qwen-turbo", "name": "qwen-turbo"}
    ]
    
    # 依次尝试每个模型
    for i, model_config in enumerate(models):
        try:
            if model_config["client"] == client_gpt:
                # GPT模型调用
                completion = model_config["client"].chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model_config["model"]
                )
            else:
                # Qwen模型调用
                completion = model_config["client"].chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model_config["model"],
                    extra_body={"enable_thinking": False},
                    response_format={"type": "text"},
                )
            
            generated_description = completion.choices[0].message.content
            
            # 线程安全地写入CSV文件
            with log_lock:
                result_row = pd.DataFrame({'index': [asin], 'description': [generated_description]})
                result_row.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            
            model_label = "主模型" if i == 0 else f"备用模型{i}"
            log_print(f"已处理第{index+1}行({model_label}-{model_config['name']}): {asin}: __ {generated_description[:100]}...")
            return {'success': True, 'index': index, 'asin': asin, 'description': generated_description}
            
        except Exception as e:
            model_label = "主模型" if i == 0 else f"备用模型{i}"
            log_print(f"{model_label}{model_config['name']}调用错误: {e}")
            
            # 如果不是最后一个模型，尝试下一个
            if i < len(models) - 1:
                next_model = models[i + 1]
                log_print(f"尝试使用备用模型{i+1}-{next_model['name']}...")
            else:
                # 所有模型都失败，标记为生成失败
                with log_lock:
                    result_row = pd.DataFrame({'index': [asin], 'description': ["生成失败"]})
                    result_row.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                
                log_print(f"第{index+1}行所有模型都失败，标记为生成失败")
                return {'success': False, 'index': index, 'asin': asin, 'description': "生成失败"}

# 准备数据用于并行处理
row_data_list = []
for index, row in df.iterrows():
    row_data_list.append((index, row['index'], row['title'], row['reviews'], row['description']))

# 使用线程池并行处理
success_count = 0
fail_count = 0
max_workers = 58  # 可以根据API限制调整并发数

log_print(f"开始并行处理 {len(row_data_list)} 条数据，使用 {max_workers} 个线程")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务
    futures = [executor.submit(process_single_row, row_data) for row_data in row_data_list]
    
    # 处理完成的任务
    for future in as_completed(futures):
        try:
            result = future.result()
            if result['success']:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            log_print(f"处理任务时发生错误: {e}")
            fail_count += 1

log_print(f"\n处理完成！结果已追加保存到 {output_file}")
log_print(f"成功处理: {success_count} 条")
log_print(f"失败处理: {fail_count} 条")
log_print(f"总计处理: {success_count + fail_count} 条")

# 关闭日志文件
log_file.close()