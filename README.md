基于BAP论文实现的个性化推荐代理系统，通过大语言模型实现智能推荐。
论文提出BAP框架（平衡采样、自适应补全、渐进优化），在"用户-代理-平台"范式下系统性解决三大问题： 平衡采样策略（BSS） 自适应描述补全（ADC） 渐进工作流程优化（PWO） 实验结果 真实数据集实验表明，BAP相比i2Agent整体性能提升26.8%，在HR、NDCG、MRR等指标上优势显著。

## 项目概述

本项目实现了一个基于LLM的推荐系统，主要功能：
- 五步推荐工作流：推荐生成 → 用户画像 → 知识提取 → 动态兴趣 → 重排序
- 支持多种数据集（Amazon Books、MovieTV、Yelp、Goodreads等）
- 仅提供动态推荐模式

## 核心文件

- **BAP.py** - 推荐代理核心实现，包含五步推荐工作流
- **Main.py** - 主执行脚本，处理数据加载和实验运行
- **.env** - API密钥配置文件
- **优化描述值/** - 商品描述优化工具集

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
在 `.env` 文件中配置：
```env
OPENAI_API_KEY='your-key'
DASHSCOPE_API_KEY='your-key'
```

### 3. 运行实验
```bash
# 基本运行
python Main.py --dataset amazon --domain books --agent_type dynamic --Description_missing _description

# 参数说明
# --dataset: 数据集 (amazon, goodreads, yelp)
# --domain: 领域 (books, movietv, yelp, reads) 
# --agent_type: 代理类型 (static, dynamic)
# --Description_missing：是否有商品描述缺失(_description=已将缺失值填入, ''=未将缺失值填入)
```

## 主要功能

### BAP.py - 推荐代理核心
- **五步工作流**：推荐内容生成 → 用户画像 → 知识生成 → 动态兴趣 → 重排序
- **评估指标**：Hit Rate、NDCG、MRR
- **错误处理**：API重试机制、JSON解析容错

### Main.py - 主执行脚本
- **数据处理**：分块处理、并发执行
- **样本采样**：正负样本交叉排列
- **结果记录**：日志文件、指标统计

### 优化描述值/ - 工具集
1. - `process_descriptions.py` - 处理描述格式
2. - `generate_descriptions.py` - 生成商品描述
3. - `merge_descriptions.py` - 合并描述数据
4. - `update_pkl_descriptions.py` - 更新数据文件

## 输出结果

实验结果保存在 `result/` 目录：
- 日志文件：详细执行过程
- 结果文件：性能指标统计
