# EVA2: Evolving Visual Adversaries for Red-Teaming GUI Agents

EVA2 is a three-stage evaluation pipeline for adversarial popup attacks against Vision-Language Model GUI agents.

Pipeline:

1. Collect successful attacks
2. Distill reusable rules
3. Evaluate rule quality with newly generated attacks

## Workflow

Collection -> Distillation -> Evaluation

## Project Structure

```text
eva2/
├── run_collection.py
├── run_distillation.py
├── run_evaluation.py
├── run_benchmark.py
├── run_local_victims.sh
├── collection/                    # Stage 1: attack collection
│   ├── run.py
│   ├── benchmark_runner.py
│   ├── attacker.py
│   ├── victim.py
│   ├── evaluator.py
│   ├── environment.py
│   ├── config.py
│   ├── tasks.json
│   └── seeds.json
├── distillation/                  # Stage 2: rule distillation
│   ├── run.py
│   ├── distiller.py
│   └── config.py
├── evaluation/                    # Stage 3: rule evaluation
│   ├── run.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── config.py
│   └── tasks.json
├── models/                        # LLM/VLM abstraction layer
│   ├── llm.py
│   └── vlm.py
├── websites/                      # target website pages
├── output/                        # distilled rules and artifacts
└──  evaluation_results/            # evaluation outputs
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n eva python=3.10
conda activate eva
pip install openai zhipuai selenium requests
```

### 2. Configuration

Update placeholders before running:

1. API keys and base URLs
   - models/llm.py
   - models/vlm.py
2. ChromeDriver path
   - collection/config.py
   - evaluation/config.py

## Stage 1: Collect Attacks

```bash
# default models: attacker=glm-4.5-flash, victim=glm-4v-flash
python run_collection.py

# enable attack evolution
python run_collection.py --evolution

# specify models and scenarios
python run_collection.py --attacker gpt-5-nano --victim gpt-4-vision-preview --scenarios amazon youtube --trials 3 --evolution

# resume from a specific task/seed
python run_collection.py --start-task T3 --start-seed S5 --evolution
```

Common flags:

- --attacker: attacker/evolution model
- --victim: victim GUI agent model
- --trials: trials per task-seed pair
- --threshold: success threshold
- --evolution: enable evolution when attacks fail
- --no-llm-judge: disable LLM intent judge
- --scenarios: run selected scenarios only
- --log: write logs to file

Outputs are created under model-specific folders:

- collection/<attacker>_<victim>/benchmark_results.jsonl
- collection/<attacker>_<victim>/successful_attacks.jsonl
- collection/<attacker>_<victim>/images/

## Stage 2: Distill Rules

```bash
# load successful_attacks.jsonl from a collection folder
python run_distillation.py --input collection/gpt-5-nano_glm-4v-flash

# custom distillation model and output file
python run_distillation.py --model gpt-5-chat --input collection/gpt-5-nano_glm-4v-flash --output output/rules_custom.json
```

Common flags:

- --input: file or directory containing successful_attacks.jsonl
- --model: LLM model for distillation
- --attacker/--victim: source model names for metadata/naming
- --output: output rule filename

Default output:

- output/rules_<attacker>_<victim>.json

## Stage 3: Evaluate Rules

```bash
# full evaluation with rules
python run_evaluation.py --rules output/rules.json

# set generator and victim models
python run_evaluation.py --rules output/rules.json --generator gpt-5-nano --victim qwen2.5-vl-7b-instruct --samples 3 --trials 3

# baseline: no rules
python run_evaluation.py --scenarios amazon --no-rules-experiment

# baseline: use seed attacks directly
python run_evaluation.py --scenarios amazon --use-seeds
```

Common flags:

- --rules: path to rules JSON
- --tasks: evaluation task file path (default: evaluation/tasks.json)
- --samples: generated attacks per rule
- --trials: trials per generated attack
- --scenarios: scenario subset
- --no-universal-rules: disable universal rules
- --no-rules-experiment: baseline without rules
- --use-seeds: baseline using collection seeds

Default output:

- evaluation_results/evaluation_report_<generator>_<victim>.json

## Model Support

Provider is auto-detected from model names:

- glm-*: Zhipu
- gpt-* / o1*: OpenAI-compatible endpoint
- qwen*: Qwen

## Ethical Use

This project is for defensive security research only. Do not use it for unauthorized or harmful activities.

## License

MIT License
