# This is a main script that tests the functionality of specific agents.
# It requires no user input.
import json
import pandas as pd
import os
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from openai import OpenAI
import openai
import logging
from pathlib import Path
import time
import random
from tqdm import tqdm
# from iAgent import iAgent
# from i2Agent import i2Agent
from BAP import i2Agent

class AverageMeter(object):
    """
    用于记录和计算性能指标的工具类
    跟踪多个指标(hit, ndcg, mrr等)的累计值和平均值
    """
    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        """更新各指标的值"""
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str) -> float:
        """获取指标的平均值"""
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str) -> None:
        """检查属性是否存在"""
        assert attr in self.totals and attr in self.counts

def return_title_ranking_list(ranked_list, title_dict, descript_dict):
    """
    将排名列表转换为字符串格式，包含每个物品的ID、标题和描述
    这个字符串会被传递给代理模型作为候选项列表
    
    根据论文 https://arxiv.org/pdf/2502.14662，该函数实现了候选物品的格式化，是推荐代理的关键输入组件。
    在论文3.1节中提到，推荐代理需要接收结构化的候选物品信息，包括ID、标题和描述，以便进行语义理解和推荐。
    
    Args:
        ranked_list: 物品ID列表，通常由传统推荐系统生成的初始排序结果
        title_dict: ID到标题的映射，提供每个物品的标题信息
        descript_dict: ID到描述的映射，提供每个物品的描述信息
    
    Returns:
        string_ranked_list: 格式化后的候选项列表字符串，每个物品包含ID、标题(限50字符)和描述(限20字符)
    """
    # 初始化空字符串，用于构建格式化的排名列表
    string_ranked_list = ""
    
    # 遍历排名列表中的每个物品ID
    for i in ranked_list:
        # 获取物品标题，确保是字符串类型
        title = title_dict[i] if isinstance(title_dict[i], str) else str(title_dict[i])
        # 获取物品描述，确保是字符串类型
        description = descript_dict[i] if isinstance(descript_dict[i], str) else str(descript_dict[i])

        # 清理描述文本：移除HTML标签并限制长度为20字符
        # 论文中提到需要对原始数据进行预处理，确保输入格式统一且简洁
        cleaned_description = re.sub(u"\\<.*?\\>", "", description)[:150]  # 移除HTML标签并限制长度

        # 将物品信息添加到排名列表字符串中，标题限制为50字符
        # 格式为"item id:{ID}, corresponding title:{标题}, description:{描述} ;"
        string_ranked_list += "item id:{}, corresponding title:{}, description:{} ;\n".format(i, title[:50], cleaned_description)
        # 以下是被注释掉的替代格式化方法
        # string_ranked_list += "item id:{},corresponding title:{}, description:{} ;".format(i,title_dict[i][:-20],re.sub(u"\\<.*?\\>", "",descript_dict[i][:-20]))
    
    # 返回完整的格式化排名列表字符串，将作为LLM代理的输入
    return string_ranked_list

def parse_response_last(text):
    """
    从LLM响应文本中解析排名列表和解释
    
    Args:
        text: LLM生成的响应文本
    
    Returns:
        ranking_list: 解析出的物品ID排名列表
        explanation_matches: 每个物品的解释
    """
    # 使用正则表达式提取排名列表
    ranking_list_pattern = re.compile(r'\[([\d,\s]+)\]')
    ranking_list_match = ranking_list_pattern.search(text)

    # 使用正则表达式提取解释
    explanation_pattern = re.compile(r'(\d+)\.\s\*\*(.*?)\*\*\s-\s(.*?)\n', re.DOTALL)
    explanation_matches = explanation_pattern.findall(text)

    # 处理排名列表
    if ranking_list_match:
        ranking_list_str = ranking_list_match.group(1)
        ranking_list = [int(x) for x in ranking_list_str.split(',')]
    else:
        ranking_list = []

    # 将解释处理为字典
    explanations = {
        int(match[0]): {
            'title': match[1].strip(),
            'description': match[2].strip()
        }
        for match in explanation_matches
    }
    return ranking_list, explanation_matches

def init_logger(log_dir: str, log_file: str) -> None:
    """
    初始化日志记录器
    
    Args:
        log_dir: 日志目录
        log_file: 日志文件名
    
    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    
    # 禁用HTTP请求日志打印和写入日志文件
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file), encoding='utf-8')
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)
    return logger


from concurrent.futures import ThreadPoolExecutor, as_completed

def main(chunk_num, df_data, logger, args, title_id_dict, descript_id_dict):
    """
    主函数：处理数据块并运行代理
    该函数实现了论文 https://arxiv.org/pdf/2502.14662 中推荐代理的主流程，包括数据预处理、负样本采样、代理调用、性能评估与结果记录。
    Args:
        chunk_num: 当前处理的数据块编号
        df_data: 包含指令和物品信息的数据块
        logger: 日志记录器
        args: 命令行参数
        title_id_dict: ID到标题的映射
        descript_id_dict: ID到描述的映射
    """
    # 1. 提取数据字段，分别获取用户指令、物品标题、描述、唯一ID、初始排名列表和用户评论
    instruction = df_data["instruction"].tolist()  # 用户指令列表
    title = df_data['title'].tolist()  # 物品标题列表
    description = df_data['description'].tolist()  # 物品描述列表
    asin = df_data['asin'].tolist()  # 物品唯一标识符列表
    ranked_lists = df_data['ranked_lists'].tolist()  # 初始排名列表
    reviewText = df_data['reviewText'].tolist()  # 用户评论列表

    # 2. 获取所有物品ID集合，用于负样本采样
    all_item_set = set(title_id_dict.keys())  # 所有物品ID集合

    # 3. 忽略所有警告信息，保证输出整洁
    warnings.filterwarnings("ignore")
    
    # 4. 初始化性能指标跟踪器，统计各类推荐指标
    stats = AverageMeter('hit1', 'hit3', 'hit5', 'ndcg3', 'ndcg5', 'mrr', 'agent_turnaround_time')

    futures = []  # 存储异步任务
    error_num = 0  # 错误计数
    right_num = 0  # 成功计数
    
    # 5. 使用线程池并行运行代理（论文实验为单线程 max_workers=1）
    with ThreadPoolExecutor(max_workers=56) as executor:
        for i in tqdm(range(len(instruction)), desc=f"Processing chunk {chunk_num}"):
            try:
                # 5.1 为当前数据生成格式化的排名列表字符串（用于输入给代理）
                rank_str_tmp = return_title_ranking_list(ranked_lists[i], title_id_dict, descript_id_dict)
            except:
                logger.info("merge rank list error in {}-th data ".format(i))
                continue

            # 5.2 获取正样本（目标物品）和负样本（随机采样非目标物品）
            positive_samples = list(set(asin[i][:-1]))  # 历史交互的物品作为正样本
            negative_samples = all_item_set - set(asin[i])  # 排除用户交互过的物品
            
            # 确定正样本数量，不超过6个
            num_positive = min(len(positive_samples), 10)
            selected_positive_samples = positive_samples[:num_positive]
            
            # 生成相同数量的负样本
            num_negative = num_positive
            selected_negative_samples = random.sample(list(negative_samples), num_negative)
            
            # 构建正样本和负样本的标题和描述列表
            positive_titles = [title_id_dict[item] for item in selected_positive_samples]
            positive_descriptions = [descript_id_dict[item] for item in selected_positive_samples]
            negative_titles = [title_id_dict[item] for item in selected_negative_samples]
            negative_descriptions = [descript_id_dict[item] for item in selected_negative_samples]
            
            # 提取用户对正样本的评价
            positive_reviews = []
            for pos_item in selected_positive_samples:
                # 在历史交互中找到对应物品的评价
                item_review = ""
                for idx, hist_item in enumerate(asin[i][:-1]):
                    if hist_item == pos_item:
                        item_review = reviewText[i][idx] if idx < len(reviewText[i]) else ""
                        break
                positive_reviews.append(item_review)

            # 5.3 构造代理输入，包括历史行为、目标物品、正负样本、排名列表等：将用户的最后一次交互作为预测目标
            task_input = {
                "instruction": instruction[i],  # 用户指令
                "title": title[i][:-1],  # 历史物品标题(除最后一个)
                "description": description[i][:-1],  # 历史物品描述(除最后一个)
                "asin": asin[i][:-1],  # 历史物品ID(除最后一个)
                "answer": asin[i][-1],  # 目标物品ID(最后一个)
                "ranked_list_str": rank_str_tmp,  # 格式化的排名列表
                "reviewText": reviewText[i][:-1],  # 用户评论(除最后一个)
                "positive_samples": selected_positive_samples,  # 正样本ID列表
                "positive_titles": positive_titles,  # 正样本标题列表
                "positive_descriptions": positive_descriptions,  # 正样本描述列表
                "positive_reviews": positive_reviews,  # 用户对正样本的评价列表
                "negative_samples": selected_negative_samples,  # 负样本ID列表
                "negative_titles": negative_titles,  # 负样本标题列表
                "negative_descriptions": negative_descriptions,  # 负样本描述列表
                "pure_ranked_list": ranked_lists[i]  # 原始排名列表
            }

            # 5.4 根据参数选择代理类型（静态/动态），论文中有静态和带动态记忆的两种代理
            if args.agent_type == "static":
                rec_agent = iAgent(task_input, logger)  # 基础版代理
            elif args.agent_type == "dynamic":
                rec_agent = i2Agent(task_input, logger)  # 增强版代理，带动态内存机制
            
            # rec_agent.run()

            # 5.5 提交代理任务到线程池，异步执行
            futures.append(executor.submit(rec_agent.run))
            # if i > 50:  # 仅处理前两个数据项(测试用，实际实验可去掉)
            #     break

        # 6. 收集并处理所有完成的任务，统计推荐指标
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Completing tasks in chunk {chunk_num}"):
            try:
                result = future.result()
                HIT_1, HIT_3, HIT_5 = result['HIT']  # 命中率指标
                NDCG_1, NDCG_3, NDCG_5 = result['NDCG']  # NDCG指标
                MRR = result['MRR']  # 平均倒数排名
                # agent_turnaround_time = result['agent_turnaround_time']
                
                # 6.1 更新统计信息（只统计有效结果）
                if HIT_1 != -1:
                    stats.update(hit1=HIT_1, hit3=HIT_3, hit5=HIT_5, ndcg3=NDCG_3, ndcg5=NDCG_5, mrr=MRR)
                    right_num += 1
                else:
                    error_num += 1
            except Exception as e:
                logger.error(f"Error in processing future: {e}")
                error_num += 1

    # 7. 记录整体结果到日志，包括成功/失败数和各类指标
    logger.info("right_num:{}   error number :{}".format(right_num,error_num))
    logger.info("chunk number:{},len of data:{},hit1:{},hit3:{},hit5:{},ndcg3:{},ndcg5:{},mrr:{}".format(chunk_num,len(df_data),stats.hit1,stats.hit3,stats.hit5,stats.ndcg3,stats.ndcg5,stats.mrr)) 
    
    # 8. 将结果写入文件，便于后续分析和论文复现
    with open('result/{}_{}/results_ours_{}.txt'.format(args.dataset,args.domain,args.agent_type),'a+') as f:  
        f.write("chunk number:{},len of data:{},hit1:{},hit3:{},hit5:{},ndcg3:{},ndcg5:{},mrr:{}\n".format(chunk_num,len(df_data),stats.hit1,stats.hit3,stats.hit5,stats.ndcg3,stats.ndcg5,stats.mrr))

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='iAgent')
    parser.add_argument('--dataset', type=str, default="amazon", help='数据集类型(amazon, goodreads, yelp等)')
    parser.add_argument('--domain', type=str, default="reads", help='领域类型(books, movietv, yelp, reads)')
    parser.add_argument('--agent_type', type=str, default="dynamic", help='代理类型(static=iAgent, dynamic=i²Agent)')
    parser.add_argument('--Description_missing', type=str, default="", help='是否有商品描述缺失(_description=已将缺失值填入, ''=未将缺失值填入)')

    args = parser.parse_args()

    # 加载数据集 - 使用在论文中构建的INSTRUCTREC数据集，包含用户指令和物品信息
    # 论文中提到的四个推荐数据集之一，来源于Amazon, Goodreads或Yelp
    df_data = pd.read_pickle("data/{}All_recagent{}.pkl".format(args.domain, args.Description_missing))
    
    # 初始化日志记录器 - 用于记录实验过程和结果
    # 日志存储在domain和dataset对应的目录下，便于后续分析
    logger = init_logger(
        "result/{}_{}/".format(args.dataset, args.domain),  # 日志目录按数据集和领域分类
        "resultsteps_{}".format(time.strftime('%m_%d_%H_%M_%S', time.localtime()))+"_{}.log".format(args.agent_type)  # 日志文件名包含时间戳和代理类型
    )
    logger.info(vars(args))  # 记录实验参数配置
    
    # 将数据集分成多个块进行处理，以便于并行处理和内存管理
    # 论文中的实验涉及大量数据，分块处理有助于提高效率
    split_size = 1000  # 每个数据块的大小
    num_chunks = len(df_data) // split_size + (1 if len(df_data) % split_size != 0 else 0)  # 计算数据块数量
    
    # 加载物品ID到标题和描述的映射 - 用于构建代理模型的知识库
    # 论文中提到代理需要物品的详细信息来理解用户指令和物品特性
    df_dict = pd.read_csv("data/combined_{}_asin_mapping{}.csv".format(args.domain, args.Description_missing))
    title_id_dict, descript_id_dict = {}, {}  # 初始化映射字典
    for i in range(len(df_dict)):
        # 构建ID到标题和描述的映射，供代理模型使用
        # 这些信息帮助代理理解物品特性，形成领域专业知识
        title_id_dict[df_dict['index'].iloc[i]] = df_dict['title'].iloc[i]
        descript_id_dict[df_dict['index'].iloc[i]] = df_dict['description'].iloc[i]
    
    # 逐块处理数据 - 对每个数据块运行代理评估
    # 论文中的实验评估了代理在不同数据集上的性能
    for i in tqdm(range(num_chunks), desc="Processing data chunks"):

        start_index = i * split_size  # 当前块的起始索引
        end_index = min(start_index + split_size, len(df_data))  # 当前块的结束索引
        df_chunk = df_data.iloc[start_index:end_index]  # 提取当前数据块
        print("len:{}".format(len(df_chunk)))  # 打印当前块的大小
        
        # 调用main函数处理当前数据块
        # 如果agent_type为static，使用基础的iAgent；如果为dynamic，使用带记忆机制的i²Agent
        # 论文中比较了这两种代理模型的性能差异
        if i > 3:
            main(i, df_chunk, logger, args, title_id_dict, descript_id_dict)