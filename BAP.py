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
from dotenv import load_dotenv

load_dotenv()

# 全局模型配置
# GLOBAL_MODEL = "qwen-turbo-latest"

GLOBAL_MODEL = "gpt-4o-mini"

# GLOBAL_MODEL = "qwen3:32b"


def cal_ndcg_hr_single(answer, ranking_list, topk=10):
    """计算单个推荐结果的评估指标：命中率(HIT)、归一化折扣累积增益(NDCG)、平均倒数排名(MRR)"""
    try:
        rank = ranking_list.index(answer)
        HIT = 0
        NDCG = 0
        MRR = 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG = 1.0 / np.log2(rank + 2.0)
            HIT = 1.0
    except ValueError:
        HIT = -1 
        NDCG = -1
        MRR = -1
    return HIT, NDCG, MRR 

class BAP():
    """智能推荐代理类，通过多步骤工作流生成个性化推荐"""
    def __init__(self, task_input, logger):
        self.task_input = task_input
        self.messages = []
        # 初始化OpenAI客户端
        # self.client = OpenAI(
        #     api_key=os.environ.get("DASHSCOPE_API_KEY"),
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        # )

        self.client = OpenAI(
            base_url="https://s.lconai.com/v1/",
            api_key=os.environ.get("LCONAI_API_GPT_KEY"),
        )

        # 定义推荐工作流：用户背景 -> 用户画像 -> 知识生成 -> 动态兴趣提取 -> 最终推荐
        self.workflow = [
            {
                "message": "",
                "tool_use": None
            },
            {
                "message": "",
                "tool_use": None
            },
            {
                "message": "",
                "tool_use": None
            },
            {
                "message": "",
                "tool_use": None
            },
            {
                "message": "",
                "tool_use": None
            }
        ]
        self.logger = logger
    
    def _log_to_file_only(self, message):
        """只写入文件，不在控制台打印的日志方法"""
        # 创建一个临时的logger，只有文件handler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.emit(self.logger.makeRecord(
                    self.logger.name, logging.INFO, "", 0, message, (), None
                ))

    def run(self):
        """执行推荐任务的主要方法"""
        max_length = 15  # 用户历史记录最大长度
        task_input = self.task_input
        # 解析任务输入数据
        instruction, title, description, asin, answer, candidate_ranked_list, pure_ranked_list = (
            task_input['instruction'], task_input['title'], task_input['description'], 
            task_input['asin'], task_input['answer'], task_input['ranked_list_str'], 
            task_input['pure_ranked_list']
        )
        reviewText = task_input["reviewText"]
        positive_samples = task_input["positive_samples"]
        positive_titles = task_input["positive_titles"]
        positive_descriptions = task_input["positive_descriptions"]
        positive_reviews = task_input["positive_reviews"]
        negative_samples = task_input["negative_samples"]
        negative_titles = task_input["negative_titles"]
        negative_descriptions = task_input["negative_descriptions"]
        
        # 截取最近的用户历史记录
        title, description, asin = title[-max_length:], description[-max_length:], asin[-max_length:]
        # 构建用户记忆字符串（包含历史交互的商品信息）
        user_memory = ""
        for j in range(len(asin)):
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            user_memory += "user historical information, item title:{},item description:{} ;".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str)
            )
        # 构建用户之前的记忆（包含评论信息）
        user_memory_previous = ""
        for j in range(len(asin)-1):
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            review_str = reviewText[j][-200:] if isinstance(reviewText[j][-200:], str) else str(reviewText[j][-200:])
            user_memory_previous += "Title:{}\nDescription:{}\nReview:{}\n\n ".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str),
                re.sub(u"\\<.*?\\>", "", review_str)
            )
        
        workflow = self.workflow

        try:
            self.messages_initial = []
            
            if workflow:
                MRR = None
                # 执行五步推荐工作流
                for i, step in enumerate(workflow):
                    message = step["message"]
                    tool_use = step["tool_use"]
                    # 步骤1：初始推荐内容生成
                    if i == 0:
                        # 构建候选物品列表字符串（交叉排列正负样本）
                        candidate_items_str = ""
                        item_counter = 1
                        
                        # 交叉排列正负样本
                        max_samples = max(len(positive_titles), len(negative_titles))
                        for idx in range(max_samples):
                            # 添加正样本（如果还有）
                            if idx < len(positive_titles):
                                pos_title, pos_desc = positive_titles[idx], positive_descriptions[idx]
                                cleaned_desc = re.sub(u"\\<.*?\\>", "", str(pos_desc)[-200:])
                                candidate_items_str += f"Item {item_counter} : title: {pos_title}, description: {cleaned_desc}\n"
                                item_counter += 1
                            
                            # 添加负样本（如果还有）
                            if idx < len(negative_titles):
                                neg_title, neg_desc = negative_titles[idx], negative_descriptions[idx]
                                cleaned_desc = re.sub(u"\\<.*?\\>", "", str(neg_desc)[-200:])
                                candidate_items_str += f"Item {item_counter} : title: {neg_title}, description: {cleaned_desc}\n"
                                item_counter += 1
                        
                        total_items = len(positive_titles) + len(negative_titles)
                        recommend_count = total_items // 2  # 推荐总样本数的一半
                        
                        step_one_message_str = "{} Here are {} candidate items for recommendation:\n{}\nPlease select {} items from the above candidates that you would recommend to this user. Please return a JSON object with a single key 'recommendations' containing a list of objects, each with 'item_number' and 'title' fields.".format(
                            message, total_items, candidate_items_str, recommend_count
                        )
                        self.messages_initial.append({
                            "role": "user",
                            "content": step_one_message_str
                        })
                        retries = 0
                        # API调用重试机制
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                    messages=self.messages_initial,
                                    model=GLOBAL_MODEL,
                                    # extra_body={"enable_thinking": False},
                                    response_format={"type": "json_object"},
                                )
                                try:
                                    response_content = completion.choices[0].message.content
                                    # 检查响应是否被markdown代码块包裹（```json...```）
                                    if response_content.strip().startswith("```json"):
                                        response_content = response_content.strip()[7:-4]
                                    recommendations = json.loads(response_content)["recommendations"]
                                    
                                    # 格式化推荐内容
                                    recommend_content = "Selected recommendations:\n"
                                    for rec in recommendations:
                                        recommend_content += f"Item {rec['item_number']}: title:{rec['title']}\n"
                                    
                                    # 将生成的推荐内容添加到消息列表
                                    self.messages_initial.append({
                                        "role": "assistant",
                                        "content": "The recommend content: {}. ".format(recommend_content)
                                    })
                                    break
                                except Exception as e:
                                    self.logger.info(f"JSON解析错误: {e}")
                                    recommend_content = "Extract Error!"
                                    retries += 1    
                            
                            except Exception as e:
                                self.logger.info(f"API调用错误: {e}")
                                recommend_content = "Extract Error!"
                                retries += 1
                                self.logger.info(f"请求超时。第 {retries} 次重试，共 {3} 次。{5} 秒后重试...")
                                time.sleep(5)

                        self._log_to_file_only("第一步消息: {} \n".format(step_one_message_str))
                        self._log_to_file_only(f"生成的推荐内容: {recommend_content}\n")
                    # 步骤2：用户画像生成
                    if i == 1:
                        # 构建正样本信息字符串
                        positive_items_info = ""
                        for idx, (pos_title, pos_review) in enumerate(zip(positive_titles, positive_reviews)):
                            # 截断评论文本到200字符
                            review_text = pos_review[-200:] if isinstance(pos_review, str) and pos_review else "No review available"
                            positive_items_info += f"Title: {pos_title}---Review: {review_text};\n "
                        
                        second_message_str = "{}. Great! Actually, this user has interacted with these items: {}. Can you generate the profile of this user background based on their preferences shown in these interactions? Please make a detailed profile. Don't use numerical numbering for the generated content; you can use bullet points instead. Please return a JSON object with a single key 'profile'.".format(message, positive_items_info)
                        self.messages_initial.append({
                            "role": "user",
                            "content": second_message_str,
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages_initial,
                                                model=GLOBAL_MODEL,
                                                # extra_body={"enable_thinking": False},
                                                response_format={"type": "json_object"},
                                            )
                                try:
                                    response_content = completion.choices[0].message.content
                                    if response_content.strip().startswith("```json"):
                                        response_content = response_content.strip()[7:-4]
                                    profile = json.loads(response_content)["profile"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"JSON解析错误: {e}")
                                    profile = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"API调用错误: {e}")
                                profile = "Extract Error!"
                                retries += 1
                                self.logger.info(f"请求超时。第 {retries} 次重试，共 {3} 次。{5} 秒后重试...")
                                time.sleep(5)

                        self._log_to_file_only("第二步消息: {}\n".format(second_message_str))
                        self._log_to_file_only(f"生成的用户画像:\n {profile}\n")
                    # 步骤3：相关知识生成
                    if i == 2:
                        third_message_str = "Please generate relevant knowledge based on the following content. You need to clearly specify the types of descriptions that should be included when recommending products, but do not directly recommend specific products. Finally, return a JSON - formatted object. This object should only contain one key 'knowledge' and its corresponding value should be a STRING. This string separates each relevant knowledge point with a newline character, and the knowledge content should be presented in the form of simple text strings. If necessary, you can use bullet points for formatting. Do not create nested objects or arrays. \n Instruction:{} \n Items previously purchased by the user and their reviews of the items:{}".format(instruction,user_memory_previous)
                        self.messages.append({
                            "role": "user",
                            "content": third_message_str
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages,
                                                model=GLOBAL_MODEL,
                                                # extra_body={"enable_thinking": False},
                                                response_format={"type": "json_object"},
                                            )
                                try:
                                    response_content = completion.choices[0].message.content
                                    if response_content.strip().startswith("```json"):
                                        response_content = response_content.strip()[7:-4]
                                    knowledge_data = json.loads(response_content)["knowledge"]

                                    # 处理嵌套的knowledge结构：API可能返回嵌套对象或直接字符串
                                    if isinstance(knowledge_data, dict) and "description_types" in knowledge_data:
                                        # 如果是嵌套结构，提取description_types列表并合并为字符串
                                        knowledge_tool_str = "\n".join(knowledge_data["description_types"])
                                    elif isinstance(knowledge_data, str):
                                        # 如果直接是字符串，直接使用
                                        knowledge_tool_str = knowledge_data
                                    else:
                                        # 其他情况转换为字符串
                                        knowledge_tool_str = str(knowledge_data)

                                    # 将生成的推荐内容添加到消息列表
                                    self.messages.append({
                                        "role": "assistant",
                                        "content": "The Knowledge: {} \n. ".format(knowledge_tool_str)
                                    })
                                    break
                                except Exception as e:
                                    self.logger.info(f"JSON解析错误: {e}")
                                    knowledge_tool_str = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"API调用错误: {e}")
                                knowledge_tool_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"请求超时。第 {retries} 次重试，共 {3} 次。{5} 秒后重试...")
                                time.sleep(5)
                        self._log_to_file_only("第三步消息: {}\n".format(third_message_str))
                        self._log_to_file_only(f"生成的知识内容: {knowledge_tool_str}\n")
                    # 步骤4：动态兴趣和动态画像生成
                    if i == 3:
                        forth_message_str = "Please use the following Knowledge and Instruction to extract the user's dynamic interests. Also, use the following Profile and Instruction to extract the user's dynamic profile information. Do not use numbers to represent generated content; use bullet points. Please return a JSON object with two keys: 'Dynamic_interest' and 'Dynamic_profile'. \n Generated Knowledge:{} Instruction:{} Profile:{}".format(knowledge_tool_str,instruction,profile)
                        self.messages.append({
                            "role": "user",
                            "content": forth_message_str,
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                # 使用完整的消息历史以保持上下文连贯性
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model=GLOBAL_MODEL,
                                    # extra_body={"enable_thinking": False},
                                    response_format={"type": "json_object"},
                                )
                                try:
                                    response_content = completion.choices[0].message.content
                                    if response_content.strip().startswith("```json"):
                                        response_content = response_content.strip()[7:-4]
                                    response_dict = json.loads(response_content)
                                    dynamic_interest_str = response_dict["Dynamic_interest"]
                                    dynamic_profile_str = response_dict["Dynamic_profile"]

                                    # 将生成的推荐内容添加到消息列表
                                    self.messages.append({
                                        "role": "assistant",
                                        "content": "The Dynamic interest: {} and the Dynamic profile: {} \n. ".format(dynamic_interest_str,dynamic_profile_str)
                                    })
                                    break
                                except Exception as e:
                                    self.logger.info(f"JSON解析错误: {e}")
                                    dynamic_interest_str = "Extract Error!"
                                    dynamic_profile_str = "Extract Error!"
                                    retries += 1   
                            except Exception as e:
                                self.logger.info(f"API调用错误: {e}")
                                dynamic_interest_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"请求超时。第 {retries} 次重试，共 {3} 次。{5} 秒后重试...")
                                time.sleep(5)
                        self._log_to_file_only("第四步消息: {}\n".format(forth_message_str))
                        self._log_to_file_only(f"动态兴趣: {dynamic_interest_str} 动态画像: {dynamic_profile_str}\n")
                    # 步骤5：最终推荐生成和重排序
                    if i == len(workflow) - 1:
                        fifth_message_str = "Based on the following information, reorganize the original recommendation list for the user: Pure Ranking List--{} \n You MUST return a JSON object with exactly two keys: 'rerank_list' (a list of integers) and 'explanation' (a list of strings). Don't use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{}, Static User Profile:{}, Dynamic User Profile:{}".format(pure_ranked_list,candidate_ranked_list,knowledge_tool_str,dynamic_interest_str,profile,dynamic_profile_str)
                        self.messages.append({
                            "role": "user",
                            "content": fifth_message_str,
                            })
                        retries = 0  # 重试计数器
                        while retries < 3:
                            try:
                                # 使用完整的消息历史以保持上下文连贯性
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model= GLOBAL_MODEL,
                                    # extra_body={"enable_thinking": False},
                                    response_format={"type": "json_object"},
                                )
                                response = completion.choices[0].message.content
                                try:
                                    response_content = response
                                    if response_content.strip().startswith("```json"):
                                        response_content = response_content.strip()[7:-4]
                                    response_dict = json.loads(response_content)
                                    rerank_list, explanation = response_dict["rerank_list"],response_dict["explanation"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"JSON解析错误: {e}")
                                    rerank_list = "Extract Error!"
                                    explanation = "Extract Error!"
                                    retries += 1
                            except Exception as e:
                                self.logger.info(f"API调用错误: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                                self.logger.info(f"请求超时。第 {retries} 次重试，共 {3} 次。{5} 秒后重试...")
                                time.sleep(5)  # 等待5秒后重试
                        self._log_to_file_only("第五步消息: {}\n".format(fifth_message_str))
                        self._log_to_file_only("纯排序列表:{}, 答案:{}, LLM排序列表:{}\n".format(pure_ranked_list,answer,rerank_list))
                        # 计算推荐效果评估指标
                        HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)   # top-1指标
                        HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)   # top-3指标
                        HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)   # top-5指标

                # 重试机制：如果MRR为-1（表示答案不在排序列表中），进行重试
                retry_mrr_times = 0
                while MRR == -1 and retry_mrr_times<3:
                    retry_mrr_times += 1
                    self.logger.info(f"生成错误的排序列表: {rerank_list}\n")
                    # 构建重试消息，要求重新排序
                    retry_message_str = "Rerank list is out of the order, you should rerank the item from the pure ranking list. The previous list:{}. Therefore, try it again according the following information.".format(rerank_list)
                    self.messages.append({
                            "role": "user",
                            "content": "{}.\n You MUST return a JSON object with exactly two keys: 'rerank_list' (a list of integers) and 'explanation' (a list of strings). Don't use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{},Static Interest:{}, User Profile:{}, Dynamic User Profile:{}, Please generate the reranked list from Pure Ranking List:{}. The length of the reranked list should be 10.".format(retry_message_str,candidate_ranked_list,knowledge_tool_str,dynamic_interest_str,user_memory_previous,profile,dynamic_profile_str,pure_ranked_list)
                            })
                    retries = 0
                    # API重试机制：最多重试3次
                    while retries < 3:
                        try:
                            # 重新调用OpenAI API生成排序列表，使用完整的消息历史以保持上下文连贯性
                            completion = self.client.chat.completions.create(
                                messages=self.messages,
                                model=GLOBAL_MODEL,
                                # extra_body={"enable_thinking": False},
                                response_format={"type": "json_object"},
                            )
                            response = completion.choices[0].message.content
                            try:
                                # 解析API响应，提取重排序列表和解释
                                response_content = response
                                if response_content.strip().startswith("```json"):
                                    response_content = response_content.strip()[7:-4]
                                response_dict = json.loads(response_content)
                                rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                break
                            except Exception as e:
                                self.logger.info(f"JSON解析错误: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                        
                        except Exception as e:
                            self.logger.info(f"API调用错误: {e}")
                            rerank_list = "Extract Error!"
                            retries += 1
                            self.logger.info(f"请求超时。第 {retries} 次重试，共 {3} 次。{5} 秒后重试...")
                            time.sleep(5)  # 等待5秒后重试
                    
                    # 记录重试后的结果并重新计算评估指标
                    self.logger.info("再次尝试，纯排序列表:{}, 答案:{}, LLM排序列表:{}\n".format(pure_ranked_list,answer,rerank_list))
                    HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                    HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                    HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                # 返回推荐效果评估指标
                return {
                    "HIT":(HIT_1,HIT_3,HIT_5),    # 命中率指标（top-1, top-3, top-5）
                    "NDCG":(NDCG_1,NDCG_3,NDCG_5), # 归一化折扣累积增益指标
                    "MRR":MRR,                     # 平均倒数排名指标
                }

            else:
                # 工作流为空时返回错误指标
                return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }
                    
        except Exception as e:
            # 异常处理：记录错误并返回失败指标
            self.logger.error(e)
            return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }
