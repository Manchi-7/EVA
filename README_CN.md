# EVA: 面向 GUI 智能体红队测试的进化式视觉对抗方法

EVA 是一个面向 Vision-Language Model GUI Agent 的对抗性弹窗评测框架。
项目提供完整的三阶段流水线：

1. 采集成功攻击样本
2. 提炼可复用攻击规则
3. 用规则生成新攻击并评估泛化效果

## 整体流程

采集(Collection) -> 蒸馏(Distillation) -> 评估(Evaluation)

## 项目结构

```text
eva2/
├── run_collection.py
├── run_distillation.py
├── run_evaluation.py
├── run_benchmark.py
├── run_local_victims.sh
├── collection/                    # 阶段1：攻击采集
│   ├── run.py
│   ├── benchmark_runner.py
│   ├── attacker.py
│   ├── victim.py
│   ├── evaluator.py
│   ├── environment.py
│   ├── config.py
│   ├── tasks.json
│   └── seeds.json
├── distillation/                  # 阶段2：规则蒸馏
│   ├── run.py
│   ├── distiller.py
│   └── config.py
├── evaluation/                    # 阶段3：规则评估
│   ├── run.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── config.py
│   └── tasks.json
├── models/                        # 模型调用层（LLM/VLM）
│   ├── llm.py
│   └── vlm.py
├── websites/                      # 测试页面
├── output/                        # 规则与中间产物
└── evaluation_results/            # 评估结果
```

## 快速开始

### 1. 环境准备

```bash
conda create -n eva python=3.10
conda activate eva
pip install openai zhipuai selenium requests
```

### 2. 基础配置

请先修改以下占位配置：

1. 模型 API Key/Base URL
   - models/llm.py
   - models/vlm.py
2. ChromeDriver 路径
   - collection/config.py
   - evaluation/config.py

## 阶段1：采集攻击数据

```bash
# 默认模型：attacker=glm-4.5-flash, victim=glm-4v-flash
python run_collection.py

# 开启攻击演化
python run_collection.py --evolution

# 指定模型和场景
python run_collection.py --attacker gpt-5-nano --victim gpt-4-vision-preview --scenarios amazon youtube --trials 3 --evolution

# 从指定任务/种子继续
python run_collection.py --start-task T3 --start-seed S5 --evolution
```

常用参数：

- --attacker: 攻击/演化模型
- --victim: 被测 GUI Agent 模型
- --trials: 每个 task-seed 的重复次数
- --threshold: 判定成功阈值
- --evolution: 启用失败后演化
- --no-llm-judge: 关闭 LLM 判定器
- --scenarios: 指定场景子集
- --log: 输出日志文件

输出目录（按模型组合自动创建）：

- collection/<attacker>_<victim>/benchmark_results.jsonl
- collection/<attacker>_<victim>/successful_attacks.jsonl
- collection/<attacker>_<victim>/images/

## 阶段2：蒸馏攻击规则

```bash
# 从 collection 子目录读取 successful_attacks.jsonl
python run_distillation.py --input collection/gpt-5-nano_glm-4v-flash

# 指定蒸馏模型与输出名
python run_distillation.py --model gpt-5-chat --input collection/gpt-5-nano_glm-4v-flash --output output/rules_custom.json
```

常用参数：

- --input: successful_attacks.jsonl 文件或所在目录
- --model: 用于规则蒸馏的 LLM
- --attacker/--victim: 用于命名与元信息（可从 input 目录名自动推断）
- --output: 输出规则文件名

默认输出：

- output/rules_<attacker>_<victim>.json

## 阶段3：评估规则质量

```bash
# 使用规则做完整评估
python run_evaluation.py --rules output/rules.json

# 指定生成器与受害者模型
python run_evaluation.py --rules output/rules.json --generator gpt-5-nano --victim qwen2.5-vl-7b-instruct --samples 3 --trials 3

# 基线：不使用规则
python run_evaluation.py --scenarios amazon --no-rules-experiment

# 基线：直接使用 seeds
python run_evaluation.py --scenarios amazon --use-seeds
```

常用参数：

- --rules: 规则文件路径
- --tasks: 评估任务文件（默认 evaluation/tasks.json）
- --samples: 每条规则生成样本数
- --trials: 每个样本试验次数
- --scenarios: 场景子集
- --no-universal-rules: 禁用通用规则
- --no-rules-experiment: 无规则基线
- --use-seeds: seeds 基线

默认输出：

- evaluation_results/evaluation_report_<generator>_<victim>.json

## 模型支持

模型提供商根据名称前缀自动识别：

- glm-*: Zhipu
- gpt-* / o1*: OpenAI 兼容接口
- qwen*: Qwen


## 安全与伦理

本项目仅用于安全研究与防御评估，请勿用于任何未授权或恶意用途。

## 许可证

MIT License
